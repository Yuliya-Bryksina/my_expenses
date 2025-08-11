# ---------- 1) Сборка фронтенда ----------
FROM node:22-alpine AS webbuild
WORKDIR /web
COPY my-expense-tracker/package*.json ./
RUN npm ci
COPY my-expense-tracker/ ./
RUN npm run build

# ---------- 2) Финальный образ: Python + Nginx + Supervisor ----------
FROM python:3.11-slim

# Системные пакеты
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx supervisor curl ca-certificates tzdata \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Podgorica
WORKDIR /app

# Копируем исходники бота и API
COPY api.py /app/api.py
COPY expense_bot.py /app/expense_bot.py

# ВАЖНО: зависимости Python
# Рекомендую сформировать requirements.txt из твоего venv:  pip freeze > requirements.txt
# Временно ставим базовые пакеты (допиши при необходимости)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Статика фронта (Vite build)
COPY --from=webbuild /web/dist /usr/share/nginx/html

# Nginx конфиг (проксируем /api → uvicorn:8000)
COPY deploy/nginx.conf /etc/nginx/nginx.conf

# Supervisor — запускаем три процесса: nginx, uvicorn, bot
COPY deploy/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Каталог для БД
RUN mkdir -p /app/data && chown -R www-data:www-data /app/data

# ENV для приложения
ENV EXPENSE_DB=/app/data/expenses.sqlite3
# BOT_TOKEN задай при запуске контейнера

EXPOSE 80

CMD ["/usr/bin/supervisord", "-n"]