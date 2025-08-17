#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expense Bot — Montenegro (single file)

Поддержка кодов: QR (pyzbar + OpenCV detectAndDecode + detectAndDecodeMulti),
DataMatrix (pylibdmtx), Aztec (zxing-cpp).
HEIC читается через pillow-heif.

Предобработка: апскейл, контраст/резкость, медианный/билатеральный фильтры,
адаптивная бинаризация, CLAHE, центральные кропы.

Если в распознанном тексте есть URL — вынимаем его.
Если URL нет, но это строка вида "iic=...&tin=..." — парсим как query.
Сохраняем все промежуточные варианты в QR_DEBUG_DIR (если задана переменная окружения).

Добавлено меню отчётов: суммы за сегодня/неделю/месяц, топ товаров/продавцов/тегов.
"""

import os
import io
import re
import json
import time
import uuid
import logging
import sqlite3
import datetime as dt
import secrets
from decimal import Decimal, InvalidOperation
from typing import Optional, Dict, Any, Iterable, List, Tuple
from datetime import datetime, date, timedelta

import requests
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from pyzbar.pyzbar import decode, ZBarSymbol
from urllib.parse import urlparse, parse_qsl

from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout
from dotenv import load_dotenv

# --- decoding watchdog ---
DECODE_TIMEOUT_SEC = int(os.environ.get("DECODE_TIMEOUT_SEC", "15"))  # seconds
_EXECUTOR = ThreadPoolExecutor(max_workers=2)

def _try_decode_with_timeout(image_bytes: bytes, timeout: int = DECODE_TIMEOUT_SEC) -> Optional[str]:
    """Run decode_any_code in a worker thread and bound the time we wait.
    Returns decoded string or None on timeout/failure.
    """
    fut = _EXECUTOR.submit(decode_any_code, image_bytes)
    try:
        return fut.result(timeout=timeout)
    except _FutTimeout:
        logger.warning("decode timeout after %ss", timeout)
        return None
    except Exception:
        logger.exception("decode failed")
        return None

# --- optional: Postgres support via psycopg ---
try:
    import psycopg
    from psycopg.rows import dict_row as _pg_dict_row
    _pg_available = True
except Exception:
    psycopg = None
    _pg_dict_row = None
    _pg_available = False


# ---------- optional backends ----------
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_ENABLED = True
except Exception:
    HEIC_ENABLED = False

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
    _cv2_available = True
except Exception:
    _cv2_available = False

try:
    from pylibdmtx.pylibdmtx import decode as dmtx_decode
    _dmtx_available = True
except Exception:
    _dmtx_available = False

try:
    import zxingcpp  # pip install zxing-cpp
    _zxing_available = True
except Exception:
    _zxing_available = False

 # Load environment variables from .env file
load_dotenv()
# ---------- config ----------
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
STATIC_API_TOKEN = os.environ.get("STATIC_API_TOKEN", "").strip()
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_URL = f"https://api.telegram.org/file/bot{BOT_TOKEN}"
VERIFY_API = "https://mapr.tax.gov.me/ic/api/verifyInvoice"
DB_PATH = os.environ.get("EXPENSE_DB", "expenses.sqlite3")
POLL_TIMEOUT = 25
DEFAULT_TAGS = ["groceries", "meat", "household", "transport", "pharmacy", "other"]

# --- Postgres URL from env ---
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# ---- debug images (optional) ----
DEBUG_DIR = os.environ.get("QR_DEBUG_DIR", "").strip()
def _dbg_save(img, tag: str):
    if not DEBUG_DIR:
        return
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(DEBUG_DIR, f"{ts}_{tag}.png")
        img.save(path, "PNG")
    except Exception as e:
        logging.getLogger("expense_bot").warning("debug save failed: %s", e)

# ---------- logger ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("expense_bot")
if HEIC_ENABLED: logger.info("HEIC support enabled via pillow_heif")
if _cv2_available: logger.info("OpenCV fallback enabled")
if _dmtx_available: logger.info("DataMatrix fallback enabled")
if _zxing_available: logger.info("ZXing (Aztec) fallback enabled")

# ---------- DB ----------
class _PgConnAdapter:
    """Adapter to make psycopg connection look like sqlite3 connection for our usage."""
    def __init__(self, pg_conn):
        self._conn = pg_conn

    # context manager compatibility: "with conn:"
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # autocommit is enabled; nothing to do
        return False

    def execute(self, sql: str, params: tuple | list = ()):  # return cursor-like with fetch* methods
        sql_conv = sql.replace("?", "%s")
        cur = self._conn.cursor(row_factory=_pg_dict_row)
        cur.execute(sql_conv, tuple(params) if params else None)
        return cur

    def close(self):
        self._conn.close()


def init_db(path: str):
    """Create and return DB connection. If DATABASE_URL is provided and psycopg is available,
    use PostgreSQL; otherwise fall back to SQLite (original behaviour).
    Returns either sqlite3.Connection or _PgConnAdapter with a .execute() API and context manager.
    """
    is_pg = bool(DATABASE_URL and _pg_available and DATABASE_URL.startswith(("postgres://", "postgresql://")))

    if is_pg:
        # Postgres connection (autocommit for simplicity)
        pg_conn = psycopg.connect(DATABASE_URL, autocommit=True)
        conn = _PgConnAdapter(pg_conn)
        # Create schema (types adjusted for PG)
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS expenses (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    batch_id TEXT NOT NULL,
                    purchased_at TEXT NOT NULL,
                    category TEXT NOT NULL,
                    amount DOUBLE PRECISION NOT NULL,
                    currency TEXT,
                    seller TEXT,
                    iic TEXT,
                    source_url TEXT,
                    tag TEXT,
                    raw_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expenses_chat_time ON expenses(chat_id, purchased_at)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_tokens (
                    chat_id INTEGER PRIMARY KEY,
                    token   TEXT    NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            # Soft migrations (ignore if exists)
            try:
                conn.execute("ALTER TABLE expenses ADD COLUMN kind TEXT DEFAULT 'expense'")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE expenses ADD COLUMN note TEXT")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE expenses ADD COLUMN source TEXT DEFAULT 'qr'")
            except Exception:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    invoice_json TEXT,
                    items_json TEXT NOT NULL
                )
                """
            )
        logger.info("DB initialized: PostgreSQL")
        return conn

    # --- SQLite fallback (original behaviour) ---
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                chat_id INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                purchased_at TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT,
                seller TEXT,
                iic TEXT,
                source_url TEXT,
                tag TEXT,
                raw_json TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expenses_chat_time ON expenses(chat_id, purchased_at)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_tokens (
                chat_id INTEGER PRIMARY KEY,
                token   TEXT    NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        # soft migrations
        try:
            conn.execute("ALTER TABLE expenses ADD COLUMN kind   TEXT DEFAULT 'expense'")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE expenses ADD COLUMN note   TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE expenses ADD COLUMN source TEXT DEFAULT 'qr'")
        except Exception:
            pass
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                chat_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                source_url TEXT NOT NULL,
                params_json TEXT NOT NULL,
                invoice_json TEXT,
                items_json TEXT NOT NULL
            )
            """
        )
    logger.info("DB initialized: SQLite at %s", path)
    return conn



# ---------- linking helpers ----------
def upsert_api_token(db: sqlite3.Connection, chat_id: int, forced_token: str | None = None) -> str:
    """Вернуть токен для chat_id. Если уже есть — отдать существующий.
    Если задан forced_token — сохранить/обновить его как постоянный.
    """
    # Если уже есть токен — вернуть его сразу (если не принудительно меняем)
    row = db.execute("SELECT token FROM api_tokens WHERE chat_id=?", (chat_id,)).fetchone()
    if row and row[0] and not forced_token:
        return row[0]

    token = forced_token or secrets.token_urlsafe(32)
    forever = "9999-12-31 23:59:59"
    with db:
        db.execute(
            "INSERT OR REPLACE INTO api_tokens(chat_id, token, expires_at) VALUES(?,?,?)",
            (chat_id, token, forever)
        )
    return token

# ---------- small report helpers ----------
def _period_bounds(kind: str) -> Tuple[str, str]:
    """Возвращает (start_iso, end_iso) для today/week/month по локальному времени."""
    today = date.today()

    if kind == "today":
        start = datetime.combine(today, datetime.min.time())
        end   = datetime.combine(today, datetime.max.time())
    elif kind == "week":
        start = datetime.combine(today - timedelta(days=today.weekday()), datetime.min.time())  # Пн
        end   = datetime.combine(start.date() + timedelta(days=6), datetime.max.time())         # Вс
    else:  # month
        first = today.replace(day=1)
        if first.month == 12:
            next_first = first.replace(year=first.year + 1, month=1, day=1)
        else:
            next_first = first.replace(month=first.month + 1, day=1)
        start = datetime.combine(first, datetime.min.time())
        end   = datetime.combine(next_first - timedelta(days=1), datetime.max.time())

    return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")

def _sum_total(db, chat_id: int, start_iso: str, end_iso: str) -> float:
    row = db.execute(
        "SELECT ROUND(SUM(amount),2) AS s FROM expenses "
        "WHERE chat_id=? AND purchased_at BETWEEN ? AND ?",
        (chat_id, start_iso, end_iso)
    ).fetchone()
    return float(row["s"] or 0)

def _sum_by_day(db, chat_id: int, start_iso: str, end_iso: str):
    return db.execute(
        "SELECT substr(purchased_at,1,10) AS d, ROUND(SUM(amount),2) AS s "
        "FROM expenses WHERE chat_id=? AND purchased_at BETWEEN ? AND ? "
        "GROUP BY d ORDER BY d",
        (chat_id, start_iso, end_iso)
    ).fetchall()

def _top_items(db, chat_id: int, start_iso: str, end_iso: str, limit=10):
    # В твоей схеме 'category' — название позиции чека
    return db.execute(
        "SELECT lower(category) AS name, ROUND(SUM(amount),2) AS s "
        "FROM expenses WHERE chat_id=? AND purchased_at BETWEEN ? AND ? "
        "GROUP BY name ORDER BY s DESC LIMIT ?",
        (chat_id, start_iso, end_iso, limit)
    ).fetchall()

def _top_sellers(db, chat_id: int, start_iso: str, end_iso: str, limit=10):
    return db.execute(
        "SELECT COALESCE(seller,'—') AS seller, ROUND(SUM(amount),2) AS s "
        "FROM expenses WHERE chat_id=? AND purchased_at BETWEEN ? AND ? "
        "GROUP BY seller ORDER BY s DESC LIMIT ?",
        (chat_id, start_iso, end_iso, limit)
    ).fetchall()

def _top_tags(db, chat_id: int, start_iso: str, end_iso: str, limit=10):
    return db.execute(
        "SELECT COALESCE(tag,'—') AS tag, ROUND(SUM(amount),2) AS s, COUNT(*) AS n "
        "FROM expenses WHERE chat_id=? AND purchased_at BETWEEN ? AND ? "
        "GROUP BY tag ORDER BY s DESC LIMIT ?",
        (chat_id, start_iso, end_iso, limit)
    ).fetchall()

def _fmt_money(x: float, currency="EUR"):
    return f"{x:.2f} {currency}"

def _report_summary(db, chat_id: int, period: str, currency="EUR") -> str:
    start_iso, end_iso = _period_bounds(period)
    total = _sum_total(db, chat_id, start_iso, end_iso)
    rows = _sum_by_day(db, chat_id, start_iso, end_iso)

    pretty = {"today": "сегодня", "week": "неделю", "month": "месяц"}.get(period, period)
    lines = [f"<b>Сумма за {pretty}</b>",
             f"Итого: <b>{_fmt_money(total, currency)}</b>"]
    if rows:
        lines.append("")
        for r in rows:
            lines.append(f"{r['d']}: {_fmt_money(float(r['s']), currency)}")
    else:
        lines.append("\nНет трат за период.")
    return "\n".join(lines)

def _report_top(db, chat_id: int, what: str, period: str, currency="EUR") -> str:
    start_iso, end_iso = _period_bounds(period)

    if what == "items":
        title = "Топ товаров"
        rows  = _top_items(db, chat_id, start_iso, end_iso, 10)
        render = [f"<b>{title} ({period})</b>"]
        if not rows:
            render.append("Нет данных за период.")
        else:
            for r in rows:
                render.append(f"• {r['name']} — {_fmt_money(float(r['s']), currency)}")
        return "\n".join(render)

    if what == "sellers":
        title = "Топ продавцов"
        rows  = _top_sellers(db, chat_id, start_iso, end_iso, 10)
        render = [f"<b>{title} ({period})</b>"]
        if not rows:
            render.append("Нет данных за период.")
        else:
            for r in rows:
                render.append(f"• {r['seller']} — {_fmt_money(float(r['s']), currency)}")
        return "\n".join(render)

    if what == "tags":
        title = "Топ тегов"
        rows  = _top_tags(db, chat_id, start_iso, end_iso, 10)
        render = [f"<b>{title} ({period})</b>"]
        if not rows:
            render.append("Нет данных за период.")
        else:
            for r in rows:
                render.append(f"• {r['tag']} — {_fmt_money(float(r['s']), currency)} ({int(r['n'])} поз.)")
        return "\n".join(render)

    return "Неизвестный отчёт."

def main_menu_keyboard():
    return {
        "inline_keyboard": [
            [
                {"text": "За сегодня", "callback_data": "sum::today"},
                {"text": "За неделю", "callback_data": "sum::week"},
                {"text": "За месяц",  "callback_data": "sum::month"},
            ],
            [
                {"text": "Топ товаров",    "callback_data": "top::items"},
                {"text": "Топ продавцов",  "callback_data": "top::sellers"},
                {"text": "Топ тегов",      "callback_data": "top::tags"},
            ],
        ]
    }

# ---------- Telegram ----------
def tg_get(method: str, **params) -> Dict[str, Any]:
    r = requests.get(f"{API_URL}/{method}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def tg_post(method: str, **params):
    r = requests.post(f"{API_URL}/{method}", json=params, timeout=30)
    if not r.ok:
        logger.error("Telegram API error %s %s: %s", r.status_code, method, r.text)
        r.raise_for_status()
    return r.json()
def tg_send_message(chat_id: int, text: str, reply_markup: Optional[Dict[str, Any]] = None, parse_mode: str = "HTML") -> None:
    try:
        tg_post("sendMessage", chat_id=chat_id, text=text, parse_mode=parse_mode, reply_markup=reply_markup or {})
    except Exception as e:
        logger.exception("sendMessage failed: %s", e)

def tg_answer_callback_query(callback_query_id: str, text: Optional[str] = None) -> None:
    try:
        tg_post("answerCallbackQuery", callback_query_id=callback_query_id, text=text or "")
    except Exception as e:
        logger.exception("answerCallbackQuery failed: %s", e)

def tg_get_file(file_id: str) -> Optional[bytes]:
    try:
        js = tg_get("getFile", file_id=file_id)
        fp = js["result"]["file_path"]
        r = requests.get(f"{FILE_URL}/{fp}", timeout=60)
        r.raise_for_status()
        return r.content
    except Exception as e:
        logger.exception("download file failed: %s", e)
        return None

# ---------- helpers: link/query extraction ----------
URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)

def pick_url_or_query(raw: str) -> str:
    """Вытащить http(s) ссылку из текстовой каши; иначе вернуть исходник (может быть 'iic=...')."""
    s = (raw or "").strip()
    m = URL_RE.search(s)
    if m:
        return m.group(0)
    return s

def extract_params_following_redirects(raw: str) -> Dict[str, str]:
    """Принимает URL ('#/verify?...') или сырую строку 'iic=...&tin=...'. Возвращает dict."""
    s = pick_url_or_query(raw)

    # Если не URL, но похоже на query
    if "://" not in s and ("iic=" in s or "tin=" in s):
        qs = s
    else:
        final_url = s
        try:
            r = requests.get(s, allow_redirects=True, timeout=15)
            if r.ok:
                final_url = r.url
        except Exception as e:
            logger.warning("Redirect follow failed: %s", e)
        p = urlparse(final_url)
        frag = p.fragment or ""
        if "/verify?" in frag:
            qs = frag.split("?", 1)[1]
        else:
            qs = p.query or frag

    params = dict(parse_qsl(qs, keep_blank_values=True))
    # нормализация
    if "crtd" in params and "dateTimeCreated" not in params:
        params["dateTimeCreated"] = params.pop("crtd").replace(" ", "+")
    if "am" in params and "amount" not in params:
        params["amount"] = params.pop("am")
    return params

# ---------- image preprocessing + decoding ----------
def decode_any_code(image_bytes: bytes) -> Optional[str]:
    """Вернуть ЛЮБОЙ текст из 2D-кода (QR/DM/Aztec) + сохранить все варианты, если QR_DEBUG_DIR задан."""
    # 1) читаем исходник
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        _dbg_save(img, "original")
    except Exception as e:
        logger.exception("PIL open failed: %s", e)
        return None

    # 2) если есть cv2 — попытаемся локализовать QR и сделать warp
    rois: List[Image.Image] = [img]  # всегда держим полный кадр как fallback
    if _cv2_available and np is not None:
        try:
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            det = cv2.QRCodeDetector()

            ok, points = det.detect(bgr)  # просто локализация
            if ok and points is not None:
                pts = points.reshape(-1, 2).astype(np.float32)
                if pts.shape[0] >= 4:
                    w = int(np.linalg.norm(pts[1] - pts[0]))
                    h = int(np.linalg.norm(pts[2] - pts[1]))
                    side = max(w, h, 200)
                    dst = np.array([[0, 0], [side-1, 0], [side-1, side-1], [0, side-1]], dtype=np.float32)
                    M = cv2.getPerspectiveTransform(pts, dst)
                    warp = cv2.warpPerspective(bgr, M, (side, side))
                    base = [warp]
                    for scale in (2, 3, 4):
                        base.append(cv2.resize(warp, (side*scale, side*scale), interpolation=cv2.INTER_CUBIC))
                    cand = []
                    for m in base:
                        gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
                        cand.append(m)
                        cand.append(gray)
                        cand.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                           cv2.THRESH_BINARY, 31, 5))
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        cand.append(clahe.apply(gray))
                    for c in cand:
                        if len(c.shape) == 2:
                            pil = Image.fromarray(c).convert("RGB")
                        else:
                            pil = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                        rois.append(pil)
                        _dbg_save(pil, "roi_cv2")
        except Exception as e:
            logger.warning("cv2 detect/warp failed: %s", e)

    # 3) подготовка PIL-вариантов
    def pil_variants(im: Image.Image) -> List[Image.Image]:
        v: List[Image.Image] = []
        gray = im.convert("L")
        v += [im, gray, ImageOps.invert(gray), ImageOps.autocontrast(gray)]
        try:
            v.append(ImageEnhance.Contrast(gray).enhance(1.8))
            v.append(ImageEnhance.Sharpness(gray).enhance(2.0))
        except Exception:
            pass
        try:
            v.append(gray.point(lambda p: 255 if p > 140 else 0))
            v.append(gray.point(lambda p: 255 if p > 120 else 0))
        except Exception:
            pass
        try:
            v.append(im.filter(ImageFilter.SHARPEN))
            v.append(im.filter(ImageFilter.MedianFilter()))
        except Exception:
            pass
        w, h = im.size
        if max(w, h) < 1200:
            for k in (2, 3):
                try:
                    v.append(im.resize((w*k, h*k), Image.NEAREST))
                except Exception:
                    pass
        return v

    def try_decode_variants(variants: List[Image.Image]) -> Optional[str]:
        # pyzbar
        for im in variants:
            try:
                res = decode(im, symbols=[ZBarSymbol.QRCODE])
            except Exception:
                res = []
            for c in res or []:
                s = (c.data or b"").decode("utf-8", "ignore").strip()
                if s:
                    logger.info("Decoded via pyzbar: %r", s)
                    return s

        # OpenCV QR
        if _cv2_available and np is not None:
            try:
                for im in variants:
                    bgr = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
                    det = cv2.QRCodeDetector()
                    data, _, _ = det.detectAndDecode(bgr)
                    if data:
                        logger.info("Decoded via cv2: %r", data)
                        return data
                    try:
                        datas, _, _ = det.detectAndDecodeMulti(bgr)
                        for d in datas or []:
                            if d:
                                logger.info("Decoded via cv2(multi): %r", d)
                                return d
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("cv2 decode failed: %s", e)

        # DataMatrix
        if _dmtx_available and np is not None:
            try:
                for im in variants:
                    arr = np.array(im.convert("RGB"))
                    try:
                        rs = dmtx_decode(arr, max_count=5, timeout=150)
                    except TypeError:
                        rs = dmtx_decode(arr)
                    for r in rs or []:
                        s = (r.data or b"").decode("utf-8", "ignore").strip()
                        if s:
                            logger.info("Decoded via DataMatrix: %r", s)
                            return s
            except Exception as e:
                logger.warning("dmtx decode failed: %s", e)

        # ZXing (Aztec/QR)
        if _zxing_available and np is not None:
            try:
                for im in variants:
                    arr = np.array(im.convert("RGB"))
                    try:
                        results = zxingcpp.read_barcodes(arr)
                    except AttributeError:
                        one = zxingcpp.read_barcode(arr)
                        results = [one] if one else []
                    for r in results or []:
                        txt = (getattr(r, "text", "") or "").strip()
                        fmt = getattr(r, "format", "unknown")
                        if txt:
                            logger.info("Decoded via ZXing (%s): %r", fmt, txt)
                            return txt
            except Exception as e:
                logger.warning("zxing decode failed: %s", e)

        return None

    # перебор ROI -> вариантов
    for roi in rois:
        variants = pil_variants(roi)
        s = try_decode_variants(variants)
        if s:
            return s

    logger.info("No code recognized after all fallbacks (with ROI).")
    return None

# ---------- verify API ----------
def verify_invoice(params: Dict[str, str]) -> Dict[str, Any]:
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "User-Agent": "Mozilla/5.0 (X11; Mac) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
    }
    r = requests.post(VERIFY_API, data=params, headers=headers, timeout=30)
    logger.info("verifyInvoice status=%s ok=%s body[0:500]=%r", r.status_code, r.ok, (r.text or "")[:500])
    r.raise_for_status()
    return r.json()

def iter_items_from_invoice_json(data: Dict[str, Any]) -> Iterable[Tuple[str, Decimal]]:
    inv = data.get("invoice") or data
    items = inv.get("items") or inv.get("invoiceItems") or inv.get("displayItems") or []
    for it in items:
        name = (it.get("name") or "").strip()
        if not name:
            continue
        price_candidates = [
            it.get("priceAfterVat"),
            it.get("totalPriceWithVat"),
            it.get("unitPriceAfterVat"),
            it.get("unitPriceWithVat"),
        ]
        qty = it.get("quantity") or 1
        amount: Optional[Decimal] = None
        for cand in price_candidates:
            if cand is None:
                continue
            try:
                amount = Decimal(str(cand))
                break
            except InvalidOperation:
                continue
        if amount is None:
            try:
                unit = Decimal(str(it.get("unitPriceAfterVat", 0)))
                amount = unit * Decimal(str(qty))
            except InvalidOperation:
                continue
        yield (name.lower(), amount)

def extract_purchase_dt_and_meta(data: Dict[str, Any], params: Dict[str, str]) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    inv = data.get("invoice") or data
    purchased_at = inv.get("issueDateTime") or inv.get("dateTimeCreated") or params.get("dateTimeCreated") or dt.datetime.utcnow().isoformat()
    currency = inv.get("currency") or inv.get("currencyCode")
    seller = None
    for key in ("sellerName", "seller", "company", "taxPayer"):
        val = inv.get(key)
        if isinstance(val, dict):
            seller = val.get("name") or seller
        elif isinstance(val, str):
            seller = val or seller
    iic = inv.get("iic") or params.get("iic")
    return purchased_at, currency, seller, iic

# ---------- bot ----------
class ExpenseBot:
    def __init__(self, token: str, db: sqlite3.Connection):
        if not token:
            raise RuntimeError("BOT_TOKEN is not set")
        self.token = token
        self.db = db
        self.offset = 0

    def run(self) -> None:
        logger.info("Bot started. Polling long updates...")
        while True:
            try:
                js = tg_get("getUpdates", timeout=POLL_TIMEOUT, offset=self.offset + 1,
                            allowed_updates=json.dumps(["message", "callback_query"]))
                if not js.get("ok"):
                    time.sleep(2); continue
                for upd in js.get("result", []):
                    self.offset = max(self.offset, int(upd["update_id"]))
                    self.process_update(upd)
            except requests.exceptions.ReadTimeout:
                continue
            except Exception as e:
                logger.exception("Polling error: %s", e)
                time.sleep(2)

    def process_update(self, upd: Dict[str, Any]) -> None:
        if "callback_query" in upd:
            return self.process_callback(upd["callback_query"])

        msg = upd.get("message") or {}
        if not msg:
            return
        chat_id = msg["chat"]["id"]
        user_id = msg["from"]["id"]

        text = msg.get("text") or ""
        if text.startswith("/"):
            return self.process_command(chat_id, user_id, text)

        if text.strip().lower().startswith("http") or ("iic=" in text and "&" in text):
            return self.handle_qr_url(text.strip(), user_id, chat_id)

        doc = msg.get("document")
        if doc and str(doc.get("mime_type", "")).startswith("image/"):
            file_id = doc["file_id"]
            data = tg_get_file(file_id)
            if not data:
                return tg_send_message(chat_id, "Не удалось скачать файл.")
            s = _try_decode_with_timeout(data)
            if not s:
                return tg_send_message(chat_id, f"Не удалось распознать код за {DECODE_TIMEOUT_SEC} сек. Пришли фото как <b>файл</b> (без сжатия) и без перекрытий на коде, либо попробуй ещё раз.")
            return self.handle_qr_url(s, user_id, chat_id)

        photos = msg.get("photo") or []
        if photos:
            file_id = photos[-1]["file_id"]
            data = tg_get_file(file_id)
            if not data:
                return tg_send_message(chat_id, "Не удалось скачать фото.")
            s = _try_decode_with_timeout(data)
            if not s:
                return tg_send_message(chat_id, f"Не удалось распознать код за {DECODE_TIMEOUT_SEC} сек или QR не содержит ссылку. Пришли фото как <b>файл</b> (без сжатия) и попробуй ещё раз.")
            return self.handle_qr_url(s, user_id, chat_id)

        return tg_send_message(chat_id, "Пришли фото/файл с QR/DM/Aztec или просто ссылку из кода.")

    def process_command(self, chat_id: int, user_id: int, text: str) -> None:
        if text.startswith("/start"):
            tg_send_message(
                chat_id,
                "Привет! Пришли фото/файл с кодом <i>(лучше как файл, без сжатия)</i> "
                "или просто ссылку из него. Я распарсю и предложу сохранить.\n\n"
                "Быстрые отчёты:",
                reply_markup=main_menu_keyboard()
            )
            return
        
        if text.startswith("/link"):
            token = upsert_api_token(self.db, chat_id, STATIC_API_TOKEN or None)
            ttl_text = "Токен постоянный."
            txt = (
                "Готово! Свяжи веб с ботом:\n"
                f"chat_id: <code>{chat_id}</code>\n"
                f"token: <code>{token}</code>\n"
                f"{ttl_text}"
            )
            tg_send_message(chat_id, txt)  # по умолчанию parse_mode="HTML"
            return
        if text.startswith("/menu"):
            tg_send_message(chat_id, "Выбери отчёт:", reply_markup=main_menu_keyboard())
            return
        if text.startswith("/help"):
            tg_send_message(chat_id, "Я принимаю фото/файлы с QR/DM/Aztec или сами ссылки.\nПосле парсинга покажу позиции и предложу сохранить.")
            return
        if text.startswith("/tag "):
            tag = text.split(" ", 1)[1].strip()
            if not tag:
                return tg_send_message(chat_id, "Укажи тег после команды: /tag <тег>")
            with self.db:
                row = self.db.execute("SELECT batch_id FROM expenses WHERE chat_id=? ORDER BY id DESC LIMIT 1",
                                      (chat_id,)).fetchone()
                if not row:
                    return tg_send_message(chat_id, "Нет последних сохранённых покупок, нечего тегировать.")
                batch_id = row["batch_id"]
                self.db.execute("UPDATE expenses SET tag=? WHERE chat_id=? AND batch_id=?", (tag, chat_id, batch_id))
            return tg_send_message(chat_id, f"Ок, проставила тег <b>{tag}</b> для последнего чека.")
        return tg_send_message(chat_id, "Неизвестная команда. /help")

    def process_callback(self, cq: Dict[str, Any]) -> None:
        cid = cq["message"]["chat"]["id"]
        data = cq.get("data") or ""
        cqid = cq["id"]

        # ----- отчёты -----
        if data.startswith("sum::"):
            _, period = data.split("::", 1)  # today|week|month
            tg_answer_callback_query(cqid)
            txt = _report_summary(self.db, cid, period, currency="EUR")
            tg_send_message(cid, txt, reply_markup=main_menu_keyboard())
            return

        if data.startswith("top::"):
            _, what = data.split("::", 1)    # items|sellers|tags
            tg_answer_callback_query(cqid)
            # по умолчанию показываем за месяц
            txt = _report_top(self.db, cid, what, period="month", currency="EUR")
            tg_send_message(cid, txt, reply_markup=main_menu_keyboard())
            return

        # ----- сохранение/отмена/теги -----
        if data.startswith("save::"):
            token = data.split("::", 1)[1]
            tg_answer_callback_query(cqid, "Сохраняю...")
            return self.confirm_and_save_pending(token, cq["from"]["id"], cid)

        if data.startswith("cancel::"):
            token = data.split("::", 1)[1]
            with self.db:
                self.db.execute("DELETE FROM pending WHERE token=?", (token,))
            tg_answer_callback_query(cqid, "Отменила.")
            return tg_send_message(cid, "Ок, не сохраняю.")

        if data.startswith("tag::"):
            _, batch_id, tag = data.split("::", 2)
            with self.db:
                self.db.execute("UPDATE expenses SET tag=? WHERE batch_id=? AND chat_id=?", (tag, batch_id, cid))
            tg_answer_callback_query(cqid, f"Тег: {tag}")
            return tg_send_message(cid, f"Готово. Для этого чека выставлен тег <b>{tag}</b>.")

        tg_answer_callback_query(cqid)

    def handle_qr_url(self, raw_text: str, user_id: int, chat_id: int) -> None:
        try:
            params = extract_params_following_redirects(raw_text)
            if not params:
                return tg_send_message(chat_id, "Не удалось извлечь параметры из ссылки/строки.")
            logger.info("POST params: %r", params)
            data = verify_invoice(params)
        except Exception as e:
            logger.exception("handle_qr_url failed: %s", e)
            return tg_send_message(chat_id, "Ошибка запроса чека. Проверь фото/ссылку и попробуй снова.")

        items = list(iter_items_from_invoice_json(data))
        if not items:
            return tg_send_message(chat_id, "Позиции не найдены в ответе на чек.")

        purchased_at, currency, seller, iic = extract_purchase_dt_and_meta(data, params)
        currency_code = currency or "EUR"
        if isinstance(currency_code, dict):
            currency_code = currency_code.get("code", "EUR")

        token = uuid.uuid4().hex
        with self.db:
            self.db.execute("""
                INSERT INTO pending(token, user_id, chat_id, created_at, source_url, params_json, invoice_json, items_json)
                VALUES(?,?,?,?,?,?,?,?)
            """, (
                token, user_id, chat_id, dt.datetime.utcnow().isoformat(),
                raw_text, json.dumps(params, ensure_ascii=False),
                json.dumps(data, ensure_ascii=False),
                json.dumps([[n, str(a)] for n, a in items], ensure_ascii=False)
            ))

        total = sum([float(a) for _, a in items])
        lines = [
            "<b>Чек распознан:</b>",
            f"Дата покупки: <code>{purchased_at}</code>",
            f"Продавец: <code>{seller or '—'}</code>",
            f"IIC: <code>{iic or '—'}</code>",
            f"Всего позиций: <b>{len(items)}</b>",
            "",
        ]
        for name, amount in items[:20]:
            lines.append(f"• {name} — {float(amount):.2f} {currency_code}")
        if len(items) > 20:
            lines.append(f"... и ещё {len(items) - 20} строк(и)")
        lines += ["", f"<b>Итого:</b> {total:.2f} {currency_code}", "", "Сохранить эти позиции?"]

        kb = {"inline_keyboard": [[
            {"text": "✅ Сохранить", "callback_data": f"save::{token}"},
            {"text": "❌ Отмена", "callback_data": f"cancel::{token}"}
        ]]}
        tg_send_message(chat_id, "\n".join(lines), reply_markup=kb)

    def confirm_and_save_pending(self, token: str, user_id: int, chat_id: int) -> None:
        row = self.db.execute("SELECT * FROM pending WHERE token=?", (token,)).fetchone()
        if not row:
            return tg_send_message(chat_id, "Не нашла черновик для сохранения. Попробуй ещё раз.")

        params = json.loads(row["params_json"])
        data = json.loads(row["invoice_json"]) if row["invoice_json"] else {}
        items = []
        for name, amount_str in json.loads(row["items_json"]):
            try:
                items.append((name, Decimal(amount_str)))
            except InvalidOperation:
                continue

        purchased_at, currency, seller, iic = extract_purchase_dt_and_meta(data, params)
        currency_code = currency or "EUR"
        if isinstance(currency_code, dict):
            currency_code = currency_code.get("code", "EUR")
        batch_id = token

        with self.db:
            for name, amount in items:
                self.db.execute("""
                    INSERT INTO expenses(user_id, chat_id, batch_id, purchased_at, category, amount, currency, seller, iic, source_url, tag, raw_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    user_id, chat_id, batch_id, purchased_at,
                    name, float(amount), currency_code, seller, iic,
                    row["source_url"], None, json.dumps(data, ensure_ascii=False)
                ))
            self.db.execute("DELETE FROM pending WHERE token=?", (token,))

        kb = {
            "inline_keyboard": [
                [{"text": t, "callback_data": f"tag::{batch_id}::{t}"} for t in DEFAULT_TAGS[:3]],
                [{"text": t, "callback_data": f"tag::{batch_id}::{t}"} for t in DEFAULT_TAGS[3:]],
            ]
        }
        tg_send_message(chat_id, "Сохранила покупки. Хочешь проставить тег для этого чека?", reply_markup=kb)
        tg_send_message(chat_id, "Или пришли свой тег командой: /tag название_тега")

# ---------- main ----------
def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is not set")
    db = init_db(DB_PATH)
    ExpenseBot(BOT_TOKEN, db).run()

if __name__ == "__main__":
    main()
