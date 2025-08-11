# api.py
import os, sqlite3
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_PATH = os.environ.get("EXPENSE_DB", "expenses.sqlite3")

app = FastAPI(title="Expenses API")

# CORS для фронта (Vite на 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # поставь конкретный origin при желании
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def auth(authorization: str = Header(None), conn=Depends(db)) -> int:
    if not authorization or " " not in authorization:
        raise HTTPException(401, "No token")
    token = authorization.split(" ", 1)[1]
    # токены у нас постоянные (expires_at = 9999-12-31), но оставим проверку на будущее
    row = conn.execute(
        "SELECT chat_id FROM api_tokens WHERE token=? AND datetime(expires_at) > datetime('now')",
        (token,)
    ).fetchone()
    if not row:
        # вдруг токен без даты (совместимость): попробуем без expires
        row = conn.execute("SELECT chat_id FROM api_tokens WHERE token=?", (token,)).fetchone()
        if not row:
            raise HTTPException(401, "Bad/expired token")
    return int(row["chat_id"])

class ExpenseIn(BaseModel):
    purchased_at: str
    category: str
    amount: float
    currency: str = "EUR"
    seller: Optional[str] = None
    tag: Optional[str] = None
    note: Optional[str] = None
    kind: str = "expense"   # 'expense' | 'savings'
    source: str = "manual"  # 'manual' | 'qr'

@app.get("/health")
def health():
    return {"ok": True, "db": DB_PATH}

@app.get("/expenses")
def list_expenses(chat_id: int, conn=Depends(db), cid: int = Depends(auth)):
    if chat_id != cid:
        raise HTTPException(403, "Forbidden")
    rows = conn.execute("""
      SELECT id, batch_id, purchased_at, category, amount, currency, seller, tag, note, kind, source, iic, source_url
      FROM expenses
      WHERE chat_id=?
      ORDER BY purchased_at DESC, id DESC
    """, (chat_id,)).fetchall()
    return [dict(r) for r in rows]

@app.post("/expenses")
def create_expense(item: ExpenseIn, conn=Depends(db), cid: int = Depends(auth)):
    with conn:
        cur = conn.execute("""
          INSERT INTO expenses(user_id, chat_id, batch_id, purchased_at, category, amount, currency,
                               seller, iic, source_url, tag, raw_json, kind, note, source)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            cid, cid, f"api-{datetime.utcnow().timestamp()}",
            item.purchased_at, item.category, item.amount, item.currency,
            item.seller, None, None, item.tag, None, item.kind, item.note, item.source
        ))
    return {"id": cur.lastrowid}

@app.delete("/expenses/{id}")
def delete_expense(id: int, conn=Depends(db), cid: int = Depends(auth)):
    row = conn.execute("SELECT chat_id FROM expenses WHERE id=?", (id,)).fetchone()
    if not row:
        raise HTTPException(404, "Not found")
    if int(row["chat_id"]) != cid:
        raise HTTPException(403, "Forbidden")
    with conn:
        conn.execute("DELETE FROM expenses WHERE id=?", (id,))
    return {"ok": True}

@app.delete("/receipts/{batch_id}")
def delete_receipt(batch_id: str, conn=Depends(db), cid: int = Depends(auth)):
    # удаляем все позиции чека атомарно
    with conn:
        conn.execute("DELETE FROM expenses WHERE chat_id=? AND batch_id=?", (cid, batch_id))
    return {"ok": True}