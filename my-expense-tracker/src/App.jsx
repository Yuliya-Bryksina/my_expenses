import React, { useEffect, useMemo, useState, useLayoutEffect } from "react";
import { motion } from "framer-motion";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts";
import { Plus, Trash2, PiggyBank, Wallet, Calendar, Settings, Moon, Sun, Edit3, Link as LinkIcon, RefreshCcw, ChevronDown, ChevronUp } from "lucide-react";

// ---------------------------------------------
// ВЕРСИЯ С ОБЩЕЙ БАЗОЙ ДАННЫХ ЧЕРЕЗ API
// - Веб и Telegram-бот делят одну SQLite БД
// - Удаление чеков, пришедших из бота, доступно из веба (DELETE /expenses/{id})
// - Ручной ввод в вебе = POST /expenses
// - Авторизация: одноразовый токен, полученный из бота командой /link
//   (токен мапится на chat_id в таблице api_tokens)
// ---------------------------------------------

// -------------------- Конфиг --------------------
// Укажи адрес API. Для локального запуска FastAPI: http://localhost:8000
const API_URL = (window.__API_URL__ || "http://localhost:8000").replace(/\/$/, "");

// -------------------- Types --------------------
/** @typedef {"expense" | "savings"} TxType */
/** @typedef {"qr" | "manual"} SourceType */

/**
 * @typedef {Object} Tx
 * @property {string} id
 * @property {number} amount
 * @property {string} category
 * @property {string} note
 * @property {string} date // ISO
 * @property {TxType} type
 * @property {SourceType} source
 * @property {string=} seller
 * @property {string=} currency
 */

/**
 * @typedef {Object} Goals
 * @property {number} weeklyBudget
 * @property {number} monthlyBudget
 * @property {number} savingsTarget
 * @property {string} savingsDeadline // ISO date (YYYY-MM-DD)
 * @property {string} currency
 */

// -------------------- Utilities --------------------
const uid = () => Math.random().toString(36).slice(2, 10);
const todayISO = () => new Date().toISOString().slice(0, 10);
const readLS = (k, fallback) => { try { const raw = localStorage.getItem(k); return raw ? JSON.parse(raw) : fallback; } catch { return fallback; } };
const writeLS = (k, v) => localStorage.setItem(k, JSON.stringify(v));
const systemPrefersDark = () => typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

const startOfWeek = (d = new Date()) => { const date = new Date(d); const day = (date.getDay() + 6) % 7; date.setHours(0,0,0,0); date.setDate(date.getDate() - day); return date; };
const endOfWeek   = (d = new Date()) => { const s = startOfWeek(d); const e = new Date(s); e.setDate(s.getDate() + 7); return e; };
const startOfMonth= (d = new Date()) => new Date(d.getFullYear(), d.getMonth(), 1);
const endOfMonth  = (d = new Date()) => new Date(d.getFullYear(), d.getMonth() + 1, 1);

const fmtMoney = (n, currency = "€") => {
  const value = Number(n) || 0;
  const hasCents = Math.round(value * 100) % 100 !== 0;
  const opts = hasCents
    ? { minimumFractionDigits: 2, maximumFractionDigits: 2 }
    : { minimumFractionDigits: 0, maximumFractionDigits: 0 };
  return `${currency}${value.toLocaleString(undefined, opts)}`;
};

const DEFAULT_GOALS = /** @type {Goals} */({
  weeklyBudget: 300,
  monthlyBudget: 1200,
  savingsTarget: 5000,
  savingsDeadline: todayISO(),
  currency: "€",
});

const CATEGORIES = ["Groceries","Dining","Transport","Housing","Utilities","Health","Entertainment","Shopping","Travel","Other"];
const PALETTE = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc949","#af7aa1","#ff9da7","#9c755f","#bab0ab"];

// -------------------- API helpers --------------------
async function apiFetch(path, { token, method = 'GET', body = null, query = {} } = {}) {
  const qs = new URLSearchParams(query).toString();
  const url = `${API_URL}${path}${qs ? `?${qs}` : ''}`;
  const res = await fetch(url, {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: body ? JSON.stringify(body) : null,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.status !== 204 ? res.json() : null;
}

// -------------------- Main App --------------------
export default function App() {
  const [goals, setGoals] = useState/** @type {[Goals, Function]} */(() => readLS("exp_goals", DEFAULT_GOALS));
  const [dark, setDark] = useState(() => { const stored = readLS("exp_dark", null); return stored === null ? systemPrefersDark() : stored; });

  // auth/link state
  const [token, setToken] = useState(() => readLS("exp_token", ""));
  const [chatId, setChatId] = useState(() => readLS("exp_chat", ""));

  // data state
  const [txs, setTxs] = useState/** @type {Tx[]} */([]);
  const [loading, setLoading] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [error, setError] = useState("");

  // --- Month navigation state ---
  const [currentMonth, setCurrentMonth] = useState(() => {
    const d = new Date();
    d.setDate(1);
    return d;
  });
  const prevMonth = () => setCurrentMonth(d => new Date(d.getFullYear(), d.getMonth() - 1, 1));
  const nextMonth = () => setCurrentMonth(d => new Date(d.getFullYear(), d.getMonth() + 1, 1));
  const monthLabel = (d) => {
    const m = d.toLocaleDateString('ru-RU', { month: 'long' });
    const month = m.charAt(0).toUpperCase() + m.slice(1); // Август
    const y = d.getFullYear();
    return `${month} ${y} г.`; // "Август 2025 г."
  };
  const mStartSel = useMemo(() => startOfMonth(currentMonth), [currentMonth]);
  const mEndSel   = useMemo(() => endOfMonth(currentMonth),   [currentMonth]);

  useEffect(() => writeLS("exp_goals", goals), [goals]);
  useEffect(() => writeLS("exp_dark", dark), [dark]);
  // Применяем тему как можно раньше до пейнта, чтобы не было мерцания
  useLayoutEffect(() => {
    const root = document.documentElement;
    root.classList.toggle('dark', !!dark);
    // помогает нативным компонентам/скроллбарам подстраиваться под тему
    root.style.colorScheme = dark ? 'dark' : 'light';
  }, [dark]);

  // initial load
  useEffect(() => { if (token && chatId) refresh(); }, [token, chatId]);

  async function refresh() {
    try {
      setLoading(true); setError("");
      const rows = await apiFetch('/expenses', { token, query: { chat_id: chatId } });
      const mapped = rows.map(r => /** @type {Tx} */({
        id: String(r.id),
        batchId: r.batch_id || r.batchId || String(r.id), // fallback для ручных
        amount: r.amount,
        category: r.category,
        note: r.note || "",
        date: new Date(r.purchased_at).toISOString(),
        type: r.kind === 'savings' ? 'savings' : 'expense',
        source: r.source || (r.iic || r.source_url ? 'qr' : 'manual'),
        seller: r.seller || '',
        currency: r.currency || goals.currency,
      }));
      setTxs(mapped);
    } catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  }

  async function addTx(tx) {
    // POST /expenses
    await apiFetch('/expenses', {
      token,
      method: 'POST',
      body: {
        purchased_at: tx.date,
        category: tx.category,
        amount: tx.amount,
        currency: goals.currency,
        seller: tx.seller || null,
        tag: null,
        note: tx.note || null,
        kind: tx.type,
        source: 'manual',
      }
    });
    await refresh();
  }

  async function deleteTx(id) {
    await apiFetch(`/expenses/${id}`, { token, method: 'DELETE' });
    setTxs(prev => prev.filter(t => t.id !== id));
  }

  async function deleteReceipt(batchId) {
    await apiFetch(`/receipts/${encodeURIComponent(batchId)}`, { token, method: 'DELETE' });
    await refresh();
  }

  // ------- Derived values -------
  const now = new Date();
  const wStart = startOfWeek(now), wEnd = endOfWeek(now);
  const mStart = startOfMonth(now), mEnd = endOfMonth(now);

  const spentWeek = useMemo(() => sumAmount(filterTx(txs, { type: "expense", from: wStart, to: wEnd })), [txs]);
  const spentMonth = useMemo(() => sumAmount(filterTx(txs, { type: "expense", from: mStartSel, to: mEndSel })), [txs, mStartSel, mEndSel]);
  const savedTotal = useMemo(() => sumAmount(txs.filter(t => t.type === "savings")), [txs]);

  const weeklyPct = pct(spentWeek, goals.weeklyBudget);
  const monthlyPct = pct(spentMonth, goals.monthlyBudget);
  const savingsPct = pct(savedTotal, goals.savingsTarget);

  const daysLeft = daysBetween(now, new Date(goals.savingsDeadline));
  const toSaveLeft = Math.max(0, goals.savingsTarget - savedTotal);
  const perDaySuggestion = daysLeft > 0 ? toSaveLeft / daysLeft : 0;

  const pieData = useMemo(() => {
    const byCategory = new Map();
    filterTx(txs, { type: "expense", from: mStartSel, to: mEndSel }).forEach(t => {
      byCategory.set(t.category, (byCategory.get(t.category) || 0) + t.amount);
    });
    return Array.from(byCategory.entries()).map(([name, value]) => ({ name, value }));
  }, [txs, mStartSel, mEndSel]);

  const lineData = useMemo(() => {
    const days = 30; const arr = [];
    for (let i = days - 1; i >= 0; i--) {
      const d = new Date(); d.setHours(0,0,0,0); d.setDate(d.getDate() - i);
      const next = new Date(d); next.setDate(d.getDate() + 1);
      const amount = sumAmount(filterTx(txs, { type: "expense", from: d, to: next }));
      arr.push({ date: d.toISOString().slice(5, 10), amount });
    }
    return arr;
  }, [txs]);

  // Группы по чекам
  const receipts = useMemo(() => {
    const map = new Map();
    for (const t of txs) {
      const key = t.batchId || t.id;
      const e = map.get(key) || {
        batchId: key,
        date: t.date,
        seller: t.seller || '',
        source: t.source,
        currency: t.currency || goals.currency,
        items: [],
        total: 0,
      };
      e.items.push(t);
      e.total += t.amount;
      // дата чека — минимальная дата позиции (на случай разницы)
      if (new Date(t.date) < new Date(e.date)) e.date = t.date;
      map.set(key, e);
    }
    return Array.from(map.values()).sort((a,b) => new Date(b.date) - new Date(a.date));
  }, [txs, goals.currency]);

  // Receipts filtered by selected month
  const receiptsMonth = useMemo(() => receipts.filter(r => {
    const dt = new Date(r.date);
    return dt >= mStartSel && dt < mEndSel;
  }), [receipts, mStartSel, mEndSel]);

  // -------- Render --------
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-950 text-slate-900 dark:text-slate-100">
      <div className="max-w-6xl mx-auto p-6">
        <header className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}>
              <div className="size-10 grid place-items-center rounded-2xl bg-slate-900 dark:bg-slate-100 text-slate-100 dark:text-slate-900 shadow">
                <PiggyBank className="size-6" />
              </div>
            </motion.div>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">My Money</h1>
              <p className="text-sm text-slate-500 dark:text-slate-400">Учёт трат, цели и накопления</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <CurrencySelect value={goals.currency} onChange={(currency) => setGoals({ ...goals, currency })} />
            <IconButton title="Тема" onClick={() => setDark(v => !v)}>
              {dark ? <Sun className="size-4"/> : <Moon className="size-4"/>}
            </IconButton>
          </div>
        </header>

        {/* AUTH / LINK */}
        {(!token || !chatId) && (
          <LinkCard onSave={(t, c) => { setToken(t); setChatId(c); writeLS('exp_token', t); writeLS('exp_chat', c); refresh(); }} />
        )}

        {/* KPI Cards */}
        {(token && chatId) && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <KpiCard icon={<Wallet className="size-5"/>} title="Неделя" value={`${fmtMoney(spentWeek, goals.currency)} / ${fmtMoney(goals.weeklyBudget, goals.currency)}`} percent={weeklyPct} subtitle={`${(100 - weeklyPct).toFixed(0)}% лимита осталось`} />
              <KpiCard
                icon={<Calendar className="size-5"/>}
                title={<div className="flex items-center gap-2">
                  <IconButton title="Предыдущий месяц" onClick={prevMonth} className="p-1 h-7 w-7 leading-none">‹</IconButton>
                  <span className="whitespace-nowrap">{monthLabel(currentMonth)}</span>
                  <IconButton title="Следующий месяц" onClick={nextMonth} className="p-1 h-7 w-7 leading-none mr-3">›</IconButton>
                </div>}
                value={`${fmtMoney(spentMonth, goals.currency)} / ${fmtMoney(goals.monthlyBudget, goals.currency)}`}
                percent={monthlyPct}
                subtitle={`${(100 - monthlyPct).toFixed(0)}% лимита осталось`}
              />
              <KpiCard icon={<PiggyBank className="size-5"/>} title="Накопления" value={`${fmtMoney(savedTotal, goals.currency)} / ${fmtMoney(goals.savingsTarget, goals.currency)}`} percent={savingsPct} subtitle={daysLeft >= 0 ? `До цели ${daysLeft} дн., по ${fmtMoney(perDaySuggestion, goals.currency)} в день` : `Дедлайн прошёл`} />
            </div>

            <Tabs>
              <TabList>
                <Tab id="add" defaultChecked>Добавить</Tab>
                <Tab id="list">Операции</Tab>
                <Tab id="analytics">Аналитика</Tab>
                <Tab id="settings">Настройки</Tab>
              </TabList>

              <TabPanel forId="add">
                <AddTxForm onSubmit={async (tx) => {
                  if (editingId) {
                    // редактирование через удаление + повторное добавление (упрощённо)
                    await deleteTx(editingId);
                    await addTx({ ...tx, source: 'manual' });
                    setEditingId(null);
                  } else {
                    await addTx({ ...tx, source: 'manual' });
                  }
                }} editing={editingId ? txs.find(t => t.id === editingId) || null : null} onCancelEdit={() => setEditingId(null)} currency={goals.currency} />
              </TabPanel>

              <TabPanel forId="list">
                <div className="flex items-center justify-between mb-2">
                  <div className="text-sm text-slate-500">{loading ? 'Загрузка…' : error ? `Ошибка: ${error}` : 'Готово'}</div>
                </div>
                <div className="flex items-center gap-2 mb-3">
                  <IconButton title="Предыдущий месяц" onClick={prevMonth} className="p-1 h-7 w-7 leading-none">‹</IconButton>
                  <span className="text-sm text-slate-600 dark:text-slate-300 whitespace-nowrap">{monthLabel(currentMonth)}</span>
                  <IconButton title="Следующий месяц" onClick={nextMonth} className="p-1 h-7 w-7 leading-none">›</IconButton>
                  <IconButton title="Обновить" onClick={refresh} className="p-1 h-7 w-7 leading-none ml-1"><RefreshCcw className="size-4"/></IconButton>
                </div>
                <ReceiptsTable
                  receipts={receiptsMonth}
                  currency={goals.currency}
                  onDeleteItem={deleteTx}
                  onEditItem={(id) => setEditingId(id)}
                  onRefresh={refresh}
                  onDeleteReceipt={deleteReceipt}
                />
              </TabPanel>

              <TabPanel forId="analytics">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <Card className="h-[360px]">
                    <CardHeader title="Расходы по категориям (месяц)" description="Доля категорий в текущем месяце" />
                    <CardBody className="h-[280px]">
                      {pieData.length === 0 ? (
                        <EmptyState label="Пока нет данных по текущему месяцу" />
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie data={pieData} dataKey="value" nameKey="name" outerRadius={110}>
                              {pieData.map((_, i) => (
                                <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                              ))}
                            </Pie>
                            <Tooltip formatter={(v) => fmtMoney(Number(v), goals.currency)} />
                          </PieChart>
                        </ResponsiveContainer>
                      )}
                    </CardBody>
                  </Card>

                  <Card className="h-[360px]">
                    <CardHeader title="Ежедневные траты (30 дней)" description="Динамика расходов по дням" />
                    <CardBody className="h-[280px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={lineData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" />
                          <YAxis />
                          <Tooltip formatter={(v) => fmtMoney(Number(v), goals.currency)} />
                          <Line type="monotone" dataKey="amount" dot={false} strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardBody>
                  </Card>
                </div>
              </TabPanel>

              <TabPanel forId="settings">
                <GoalsForm goals={goals} onChange={setGoals} />
              </TabPanel>
            </Tabs>
          </>
        )}

        <footer className="mt-8 text-center text-xs text-slate-500 dark:text-slate-400">
          Данные берутся из общей БД через API • Удаление доступно и для чеков, пришедших из бота
        </footer>
      </div>
    </div>
  );
}

// -------------------- Primitive UI --------------------
function IconButton({ children, onClick, title, className = "" }) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={`inline-flex items-center justify-center rounded-2xl border border-slate-200 dark:border-slate-700 p-2 hover:bg-slate-50 dark:hover:bg-slate-800 transition ${className}`}
    >
      {children}
    </button>
  );
}

function Card({ children, className = "" }) { return <div className={`rounded-2xl border border-slate-200 dark:border-slate-800 bg-white/70 dark:bg-slate-900/60 shadow-sm ${className}`}>{children}</div>; }
function CardHeader({ title, description }) { return (<div className="px-4 pt-4 pb-2"><div className="text-base font-semibold">{title}</div>{description && <div className="text-sm text-slate-500 dark:text-slate-400">{description}</div>}</div>); }
function CardBody({ children, className = "" }) { return <div className={`px-4 pb-4 ${className}`}>{children}</div>; }
function Input({ ...props }) { return <input {...props} className={`w-full rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 py-2 outline-none focus:ring-2 focus:ring-slate-400 ${props.className||""}`} /> }
function Label({ children }) { return <label className="text-sm text-slate-600 dark:text-slate-300">{children}</label>; }
function Select({ value, onChange, children }) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full appearance-none rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-3 pr-10 py-2"
      >
        {children}
      </select>
      <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-slate-500 dark:text-slate-400">
        <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
          <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 111.06 1.06l-4.24 4.24a.75.75 0 01-1.06 0L5.21 8.29a.75.75 0 01.02-1.08z" clipRule="evenodd" />
        </svg>
      </span>
    </div>
  );
}
function Button({ children, onClick, variant="primary" }) { const base = "px-4 py-2 rounded-xl text-sm font-medium transition"; const styles = variant === "ghost" ? "bg-transparent hover:bg-slate-100 dark:hover:bg-slate-800 border border-slate-300 dark:border-slate-700" : "bg-slate-900 text-white hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white"; return <button onClick={onClick} className={`${base} ${styles}`}>{children}</button>; }
function Progress({ value }) { return (<div className="w-full h-2 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden"><div className="h-2 bg-slate-900 dark:bg-slate-100" style={{ width: `${Math.min(100, Math.max(0, value))}%` }} /></div>); }

// Tabs without external libs
function Tabs({ children }) { return <div>{children}</div>; }
function TabList({ children }) { return <div className="grid grid-cols-4 rounded-2xl border border-slate-200 dark:border-slate-800 overflow-hidden">{children}</div>; }
function Tab({ id, defaultChecked, children }) { return (<label className="cursor-pointer"><input type="radio" name="tabs" id={`tab-${id}`} defaultChecked={defaultChecked} className="peer hidden" /><span className="block px-4 py-2 text-center text-sm peer-checked:bg-slate-900 peer-checked:text-white dark:peer-checked:bg-slate-100 dark:peer-checked:text-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800">{children}</span></label>); }
function TabPanel({ forId, children }) { const [active, setActive] = useState(() => document?.querySelector(`#tab-${forId}`)?.checked ?? false); useEffect(() => { const handler = () => setActive(document?.querySelector(`#tab-${forId}`)?.checked ?? false); const radios = document.querySelectorAll('input[name="tabs"]'); radios.forEach(r => r.addEventListener('change', handler)); handler(); return () => radios.forEach(r => r.removeEventListener('change', handler)); }, [forId]); return <div className={active ? "mt-4" : "hidden"}>{children}</div>; }

// -------------------- App Components --------------------

function ReceiptsTable({ receipts, onDeleteItem, onEditItem, currency, onRefresh, onDeleteReceipt }) {
  const [open, setOpen] = useState({}); // {batchId: boolean}
  const toggle = (id) => setOpen(o => ({ ...o, [id]: !o[id] }));

  // Определяем тип чека: все чеки (особенно из QR) считаем расходами.
  // Накопления — только если это ручные позиции и все они типа 'savings'.
  const getReceiptType = (r) => {
    if (r.source === 'qr') return 'expense';
    const allSavings = r.items.length > 0 && r.items.every(it => it.type === 'savings');
    return allSavings ? 'savings' : 'expense';
  };

  const TypePill = ({ kind }) => {
    if (kind === 'savings') return (
      <span className="inline-flex items-center justify-center shrink-0 whitespace-nowrap px-2 py-0.5 rounded text-xs bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300">Накопления</span>
    );
    if (kind === 'expense') return (
      <span className="inline-flex items-center justify-center shrink-0 whitespace-nowrap px-2 py-0.5 rounded text-xs bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300">Расход</span>
    );
    return (
      <span className="inline-flex items-center justify-center shrink-0 whitespace-nowrap px-2 py-0.5 rounded text-xs bg-slate-100 text-slate-700 dark:bg-slate-800/50 dark:text-slate-300">Смешанный</span>
    );
  };

  // Replace previous deleteReceipt helper with the new wrapper
  async function deleteReceiptLocal(batchId) {
    if (!confirm('Удалить весь чек целиком? Это удалит все позиции.')) return;
    if (onDeleteReceipt) {
      await onDeleteReceipt(batchId);
      return;
    }
    const r = receipts.find(x => x.batchId === batchId);
    if (!r) return;
    for (const item of r.items) {
      await onDeleteItem(item.id);
    }
    onRefresh?.();
  }

  return (
    <Card>
      <CardHeader title="Список чеков" description="Разверните чек, чтобы увидеть позиции" />
      <CardBody>
        <div className="overflow-x-auto">
          <table className="w-full text-xs sm:text-sm">
            <thead className="text-left text-slate-500">
              <tr>
                <th className="py-2 pr-3">Дата</th>
                <th className="py-2 pr-3">Продавец</th>
                <th className="py-2 pr-3">Источник</th>
                <th className="py-2 pr-3 text-right">Итого</th>
                <th className="py-2 pl-3 pr-4 text-right">Действия</th>
              </tr>
            </thead>
            <tbody>
              {receipts.map(r => (
                <React.Fragment key={r.batchId}>
                  {/* основная строка чека */}
                  <tr className="border-t border-slate-200/60 dark:border-slate-800/60">
                    <td className="py-2 pr-3 tabular-nums">{r.date.slice(0,10)}</td>
                    <td className="py-2 pr-3">{r.seller || '—'}</td>
                    <td className="py-2 pr-3">
                      <span className={`px-2 py-1 rounded text-xs ${r.source === 'qr' ? 'bg-sky-100 text-sky-800 dark:bg-sky-900/30 dark:text-sky-300' : 'bg-violet-100 text-violet-800 dark:bg-violet-900/30 dark:text-violet-300'}`}>
                        {r.source === 'qr' ? 'QR' : 'Ручной'}
                      </span> 
                    </td>
                    <td className="py-2 pr-3 text-right">
                      {(() => {
                        const kind = getReceiptType(r);
                        const sign = kind === 'savings' ? '+' : '−';
                        const cls = kind === 'savings' ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
                        return (
                          <div className="grid grid-cols-[auto,minmax(6.5rem,auto)] items-center justify-end gap-2">
                            <TypePill kind={kind} />
                            <span className={`font-medium tabular-nums text-right ${cls}`}>{sign}{fmtMoney(r.total, currency)}</span>
                          </div>
                        );
                      })()}
                    </td>
                    <td className="py-2 pl-3 pr-4 text-right">
                      {/* Desktop / tablet: иконки */}
                      <div className="hidden sm:inline-flex items-center gap-2 justify-end">
                        <IconButton
                          title={open[r.batchId] ? "Свернуть чек" : "Развернуть чек"}
                          onClick={() => toggle(r.batchId)}
                          className="p-1 h-8 w-8"
                        >
                          {open[r.batchId] ? <ChevronUp className="size-4"/> : <ChevronDown className="size-4"/>}
                        </IconButton>
                        <IconButton
                          title="Удалить чек"
                          onClick={() => deleteReceiptLocal(r.batchId)}
                          className="p-1 h-8 w-8"
                        >
                          <Trash2 className="size-4"/>
                        </IconButton>
                      </div>
                      {/* Mobile: иконки */}
                      <div className="inline-flex sm:hidden items-center gap-1 justify-end">
                        <IconButton title={open[r.batchId] ? 'Свернуть чек' : 'Развернуть чек'} onClick={() => toggle(r.batchId)} className="p-1 h-8 w-8">
                          {open[r.batchId] ? <ChevronUp className="size-4"/> : <ChevronDown className="size-4"/>}
                        </IconButton>
                        <IconButton title="Удалить чек" onClick={() => deleteReceiptLocal(r.batchId)} className="p-1 h-8 w-8">
                          <Trash2 className="size-4"/>
                        </IconButton>
                      </div>
                    </td>
                  </tr>
                  {/* раскрывающиеся позиции */}
                  {open[r.batchId] && (
                    <tr className="bg-slate-50/60 dark:bg-slate-900/30">
                      <td colSpan={5} className="py-3 px-4">
                        <div className="overflow-x-auto">
                          <table className="w-full text-sm">
                            <thead className="text-left text-slate-500">
                              <tr>
                                <th className="py-1 pr-3">Категория</th>
                                <th className="py-1 pr-3">Заметка</th>
                                <th className="py-1 pr-3 text-right">Сумма</th>
                                <th className="py-1 pl-3">Действия</th>
                              </tr>
                            </thead>
                            <tbody>
                              {r.items.map(t => (
                                <tr key={t.id} className="border-t border-slate-200/60 dark:border-slate-800/60">
                                  <td className="py-1 pr-3">{t.category}</td>
                                  <td className="py-1 pr-3">{t.note || '—'}</td>
                                  <td className="py-1 pr-3 text-right">
                                    {(() => {
                                      const kind = t.type === 'savings' ? 'savings' : 'expense';
                                      const sign = kind === 'savings' ? '+' : '−';
                                      const cls = kind === 'savings' ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
                                      return <span className={`inline-block tabular-nums text-right min-w-[6.5rem] ${cls}`}>{sign}{fmtMoney(t.amount, currency || t.currency)}</span>;
                                    })()}
                                  </td>
                                  <td className="py-1 pl-3">
                                    <div className="flex items-center gap-1">
                                      <IconButton title="Редактировать" onClick={() => onEditItem(t.id)}><Edit3 className="size-4"/></IconButton>
                                      <IconButton title="Удалить позицию" onClick={async () => { await onDeleteItem(t.id); onRefresh?.(); }}><Trash2 className="size-4"/></IconButton>
                                    </div>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
              {receipts.length === 0 && (
                <tr><td colSpan={5} className="py-8 text-center text-slate-500"><EmptyState label="Чеков пока нет" /></td></tr>
              )}
            </tbody>
          </table>
        </div>
      </CardBody>
    </Card>
  );
}
function KpiCard({ icon, title, value, percent, subtitle }) {
  // value приходит в формате "€86,08 / €1 200,00"
  // Разделим строку по "/"
  let before = value, after = null;
  if (typeof value === "string" && value.includes("/")) {
    const parts = value.split("/");
    before = parts[0].trim();
    after = parts.slice(1).join("/").trim();
  }
  return (
    <Card>
      <div className="px-4 pt-4">
        <div className="flex items-center justify-between">
          <div className="text-base font-semibold flex items-center gap-2">{icon}{title}</div>
          <span className="text-sm font-medium tabular-nums text-right">
            {before}
            {after && (
              <span className="block">{`/ ${after}`}</span>
            )}
          </span>
        </div>
        <div className="text-sm text-slate-500 dark:text-slate-400">{subtitle}</div>
      </div>
      <div className="px-4 pb-4"><Progress value={percent} /></div>
    </Card>
  );
}

function AddTxForm({ onSubmit, editing, onCancelEdit, currency }) {
  const [amount, setAmount] = useState(editing ? String(editing.amount) : "");
  const [category, setCategory] = useState(editing ? editing.category : CATEGORIES[0]);
  const [note, setNote] = useState(editing ? editing.note : "");
  const [date, setDate] = useState(editing ? editing.date.slice(0,10) : todayISO());
  const [type, setType] = useState(editing ? editing.type : /** @type {TxType} */("expense"));

  useEffect(() => { if (editing) { setAmount(String(editing.amount)); setCategory(editing.category); setNote(editing.note); setDate(editing.date.slice(0,10)); setType(editing.type); } }, [editing?.id]);

  return (
    <Card>
      <CardHeader title={editing ? "Редактировать операцию" : "Новая операция"} description="Добавьте расход или пополнение накоплений" />
      <CardBody>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2"><Label>Тип</Label><Select value={type} onChange={(v) => setType(/** @type {TxType} */(v))}><option value="expense">Расход</option><option value="savings">Накопления</option></Select></div>
          <div className="space-y-2"><Label>Сумма ({currency})</Label><Input inputMode="decimal" type="number" step="0.01" value={amount} onChange={(e) => setAmount(e.target.value)} placeholder="0.00" /></div>
          <div className="space-y-2"><Label>Категория</Label><Select value={category} onChange={setCategory}>{CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}</Select></div>
          <div className="space-y-2 md:col-span-2"><Label>Заметка</Label><Input value={note} onChange={(e) => setNote(e.target.value)} placeholder="Например: кофе, проездной..." /></div>
          <div className="space-y-2"><Label>Дата</Label><Input type="date" value={date} onChange={(e) => setDate(e.target.value)} /></div>
          <div className="md:col-span-3 flex items-center justify-end gap-2">{editing && <Button variant="ghost" onClick={onCancelEdit}>Отмена</Button>}<Button onClick={() => { const n = Number(amount); if (!n || n <= 0) return alert("Введите корректную сумму"); /** @type {Tx} */ const tx = { id: "", amount: round2(n), category, note, date: new Date(date).toISOString(), type, source: 'manual' }; onSubmit(tx); if (!editing) { setAmount(""); setNote(""); } }}>{editing ? "Сохранить" : "Добавить"}</Button></div>
        </div>
      </CardBody>
    </Card>
  );
}

function TransactionsTable({ txs, onDelete, onEdit, currency }) {
  const [q, setQ] = useState("");
  const filtered = useMemo(() => txs.filter(t => [t.category, t.note, t.type, t.date.slice(0,10), t.source].join(" ").toLowerCase().includes(q.toLowerCase())), [txs, q]);

  return (
    <Card>
      <CardHeader title="Список операций" description="Поиск, редактирование, удаление" />
      <CardBody>
        <div className="flex flex-col md:flex-row md:items-center gap-3 mb-3">
          <Input placeholder="Поиск..." value={q} onChange={(e) => setQ(e.target.value)} className="md:max-w-xs" />
          <div className="text-xs text-slate-500">Всего: {txs.length}</div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-slate-500">
              <tr>
                <th className="py-2 pr-3">Тип</th>
                <th className="py-2 pr-3">Источник</th>
                <th className="py-2 pr-3">Дата</th>
                <th className="py-2 pr-3">Категория</th>
                <th className="py-2 pr-3">Заметка</th>
                <th className="py-2 pr-3 text-right">Сумма</th>
                <th className="py-2 pl-3">Действия</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(t => (
                <tr key={t.id} className="border-t border-slate-200/60 dark:border-slate-800/60">
                  <td className="py-2 pr-3"><span className={`px-2 py-1 rounded text-xs ${t.type === "expense" ? "bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300" : "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300"}`}>{t.type === "expense" ? "Расход" : "Накопления"}</span></td>
                  <td className="py-2 pr-3"><span className={`px-2 py-1 rounded text-xs ${t.source === 'qr' ? 'bg-sky-100 text-sky-800 dark:bg-sky-900/30 dark:text-sky-300' : 'bg-violet-100 text-violet-800 dark:bg-violet-900/30 dark:text-violet-300'}`}>{t.source === 'qr' ? 'QR' : 'Ручной'}</span></td>
                  <td className="py-2 pr-3 tabular-nums">{t.date.slice(0,10)}</td>
                  <td className="py-2 pr-3">{t.category}</td>
                  <td className="py-2 pr-3">{t.note}</td>
                  <td className="py-2 pr-3 text-right font-medium">{fmtMoney(t.amount, currency || t.currency)}</td>
                  <td className="py-2 pl-3">
                    <div className="flex items-center gap-1">
                      <IconButton title="Редактировать" onClick={() => onEdit(t.id)}><Edit3 className="size-4"/></IconButton>
                      <IconButton title="Удалить" onClick={() => onDelete(t.id)}><Trash2 className="size-4"/></IconButton>
                    </div>
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr><td colSpan={7} className="py-8 text-center text-slate-500"><EmptyState label="Нет операций" /></td></tr>
              )}
            </tbody>
          </table>
        </div>
      </CardBody>
    </Card>
  );
}

function GoalsForm({ goals, onChange }) {
  const [loc, setLoc] = useState(goals);
  useEffect(() => setLoc(goals), [goals]);
  // Для input-ов: недельный, месячный лимит, цель накоплений
  // Удаление ведущих нулей, при очистке — пусто
  const handleWeeklyLimitChange = (e) => {
    const value = e.target.value.replace(/^0+(?=\d)/, '');
    setLoc({ ...loc, weeklyBudget: value === '' ? '' : value });
  };
  const handleMonthlyLimitChange = (e) => {
    const value = e.target.value.replace(/^0+(?=\d)/, '');
    setLoc({ ...loc, monthlyBudget: value === '' ? '' : value });
  };
  const handleSavingsTargetChange = (e) => {
    const value = e.target.value.replace(/^0+(?=\d)/, '');
    setLoc({ ...loc, savingsTarget: value === '' ? '' : value });
  };
  return (
    <Card>
      <CardHeader title="Цели и бюджеты" description="Задайте недельный/месячный лимиты и цель по накоплениям" />
      <CardBody>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Недельный лимит</Label>
            <Input
              type="number"
              step="0.01"
              value={loc.weeklyBudget === 0 ? '' : loc.weeklyBudget}
              onChange={handleWeeklyLimitChange}
            />
          </div>
          <div className="space-y-2">
            <Label>Месячный лимит</Label>
            <Input
              type="number"
              step="0.01"
              value={loc.monthlyBudget === 0 ? '' : loc.monthlyBudget}
              onChange={handleMonthlyLimitChange}
            />
          </div>
          <div className="space-y-2">
            <Label>Цель накоплений</Label>
            <Input
              type="number"
              step="0.01"
              value={loc.savingsTarget === 0 ? '' : loc.savingsTarget}
              onChange={handleSavingsTargetChange}
            />
          </div>
          <div className="space-y-2"><Label>Дедлайн накоплений</Label><Input type="date" value={loc.savingsDeadline} onChange={(e) => setLoc({ ...loc, savingsDeadline: e.target.value })} /></div>
          <div className="space-y-2"><Label>Валюта</Label><CurrencySelect value={loc.currency} onChange={(currency) => setLoc({ ...loc, currency })} /></div>
          <div className="md:col-span-2 flex items-center justify-end gap-2">
            <Button variant="ghost" onClick={() => setLoc(goals)}>Сбросить</Button>
            <Button
              onClick={() =>
                onChange({
                  ...loc,
                  weeklyBudget: loc.weeklyBudget === '' ? 0 : num(loc.weeklyBudget),
                  monthlyBudget: loc.monthlyBudget === '' ? 0 : num(loc.monthlyBudget),
                  savingsTarget: loc.savingsTarget === '' ? 0 : num(loc.savingsTarget),
                })
              }
            >
              Сохранить
            </Button>
          </div>
        </div>
      </CardBody>
    </Card>
  );
}

function CurrencySelect({ value, onChange }) {
  return (
    <Select value={value} onChange={onChange}>
      <option value="€">€ Euro</option>
      <option value="$">$ USD</option>
      <option value="£">£ GBP</option>
      <option value="₽">₽ RUB</option>
      <option value="NZ$">NZ$ NZD</option>
      <option value="RSD">RSD</option>
    </Select>
  );
}

function LinkCard({ onSave }) {
  const [t, setT] = useState("");
  const [c, setC] = useState("");
  return (
    <Card className="mb-6">
      <CardHeader title="Связать веб с ботом" description="В боте отправь /link — получишь одноразовый токен и свой chat_id. Вставь их сюда, чтобы веб видел чеки и мог их удалять." />
      <CardBody>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2"><Label>Токен</Label><Input value={t} onChange={(e)=>setT(e.target.value)} placeholder="например, eyJhbGciOiJI..." /></div>
          <div className="space-y-2"><Label>chat_id</Label><Input value={c} onChange={(e)=>setC(e.target.value)} placeholder="например, 123456789" /></div>
          <div className="flex items-end"><Button onClick={() => onSave(t.trim(), c.trim())}><LinkIcon className="inline mr-2 size-4"/>Связать</Button></div>
        </div>
      </CardBody>
    </Card>
  );
}

function EmptyState({ label }) { return <div className="text-sm text-slate-500">{label}</div>; }

// -------------------- Helpers --------------------
function filterTx(txs, { type, from, to }) { return txs.filter(t => ((!type || t.type === type) && (!from || new Date(t.date) >= from) && (!to || new Date(t.date) < to))); }
function sumAmount(list) { return round2(list.reduce((s, t) => s + t.amount, 0)); }
function pct(part, total) { return total > 0 ? Math.min(100, Math.round((part / total) * 100)) : 0; }
function round2(n) { return Math.round(n * 100) / 100; }
function num(v) { const n = Number(v); return Number.isFinite(n) ? n : 0; }
function daysBetween(a, b) { const ms = Math.floor((new Date(b).setHours(0,0,0,0) - new Date(a).setHours(0,0,0,0)) / 86400000); return ms; }
