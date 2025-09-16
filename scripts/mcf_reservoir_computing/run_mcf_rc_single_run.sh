#!/usr/bin/env bash
set -Eeuo pipefail

# === Базовые пути ===
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# === Настройки (переопределяются через env) ===
SESSION_BASE="${SESSION_BASE:-mcf_rc}"                 # базовый префикс имени сессии
SESSION_AUTONAME="${SESSION_AUTONAME:-timestamp}"      # timestamp | index
PROJECT_DIR="${PROJECT_DIR:-$REPO_ROOT}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
REQ_FILE="${REQ_FILE:-$PROJECT_DIR/requirements.txt}"
MAIN_PY="${MAIN_PY:-$SCRIPT_DIR/mcf_reservoir_computing.py}"
PY_ARGS="${PY_ARGS:-}"

# Отключаем внутренний трединг BLAS/OMP/Numba
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

LOG_DIR="$SCRIPT_DIR/logs"; mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# >>> мини-правка: все служебные файлы — рядом со скриптом
STATE_DIR="$SCRIPT_DIR/.runner"
mkdir -p "$STATE_DIR"
LAST_SESSION_FILE="$STATE_DIR/.last_session"
# <<<

ensure_tmux(){ command -v tmux >/dev/null || { echo "tmux не установлен"; exit 1; }; }

bootstrap() {
  echo "[bootstrap] Проект: $PROJECT_DIR"
  mkdir -p "$PROJECT_DIR"
  cd "$PROJECT_DIR"
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[bootstrap] Создаю venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip >/dev/null
  if [[ -f "$REQ_FILE" ]]; then
    echo "[bootstrap] Устанавливаю зависимости из $REQ_FILE"
    pip install -r "$REQ_FILE"
  else
    echo "[bootstrap] $REQ_FILE не найден — пропускаю установку зависимостей."
  fi
}

# --- генерация уникального имени сессии ---
_unique_session_name() {
  local base="$1"
  echo "${base}_$(date +%Y%m%d-%H%M%S)"
}

start() {
  ensure_tmux
  bootstrap

  # всегда создаём НОВУЮ сессию с уникальным именем
  local SESSION
  SESSION="$(unique_session_name "$SESSION_BASE")"
  echo "$SESSION" > "$LAST_SESSION_FILE"

  tmux new-session -d -s "$SESSION" -n run
  tmux send-keys -t "$SESSION:run" "cd '$PROJECT_DIR'" C-m
  tmux send-keys -t "$SESSION:run" "source '$VENV_DIR/bin/activate' || true" C-m
  tmux send-keys -t "$SESSION:run" "echo \"[run] PY_ARGS: $PY_ARGS\" | tee -a '$RUN_LOG'" C-m
  tmux send-keys -t "$SESSION:run" "python '$MAIN_PY' $PY_ARGS 2>&1 | tee -a '$RUN_LOG'" C-m

  echo
  echo "✓ Запущено в tmux-сессии: $SESSION"
  echo "  Лог: $RUN_LOG"
  echo "Подсказки: $0 attach | $0 status | $0 stop"
}

# resume: если указали SESSION вручную — attach к нему; иначе к последней, созданной этим скриптом
resume() {
  ensure_tmux
  local SESSION="${SESSION:-}"
  if [[ -z "$SESSION" && -f "$LAST_SESSION_FILE" ]]; then
    SESSION="$(< "$LAST_SESSION_FILE")"
  fi
  if [[ -z "$SESSION" ]]; then
    echo "Не знаю, к какой сессии присоединяться (нет .last_session). Запусти $0 start."
    exit 1
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux attach -t "$SESSION"
  else
    echo "Сессия $SESSION не найдена."
    exit 1
  fi
}

attach() {
  ensure_tmux
  local SESSION="${SESSION:-}"
  if [[ -z "$SESSION" && -f "$LAST_SESSION_FILE" ]]; then
    SESSION="$(< "$LAST_SESSION_FILE")"
  fi
  if [[ -z "$SESSION" ]]; then
    echo "Не задано имя SESSION и нет .last_session. Укажи SESSION=... $0 attach"
    exit 1
  fi
  tmux attach -t "$SESSION"
}

status(){
  local SES="${SESSION:-}"
  if [[ -z "$SES" && -f "$SESSION_FILE" ]]; then
    SES="$(< "$SESSION_FILE")"
  fi

  if tmux list-sessions >/dev/null 2>&1; then
    echo "Активные tmux-сессии:"
    tmux list-sessions -F '#S: #{session_windows} windows, attached=#{session_attached}'
    if [[ -f "$SESSION_FILE" ]]; then
      echo "Последняя, созданная этим скриптом: $(< "$SESSION_FILE")"
    fi
    echo "Последние логи:"
    ls -1t "$LOG_DIR" | sed "s|^|$LOG_DIR/|"
  else
    echo "tmux-сервер не запущен (сессий нет)."
  fi
}

stop(){
  ensure_tmux
  local SESSION="${SESSION:-}"
  if [[ -z "$SESSION" && -f "$LAST_SESSION_FILE" ]]; then
    SESSION="$(< "$LAST_SESSION_FILE")"
  fi
  if [[ -z "$SESSION" ]]; then
    echo "Не задано имя SESSION и нет .last_session. Укажи SESSION=... $0 stop"
    exit 1
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Останавливаю сессию $SESSION…"
    tmux send-keys -t "$SESSION" C-c
    sleep 1 || true
    tmux kill-session -t "$SESSION" || true
    echo "✓ Остановлено."
  else
    echo "Сессия $SESSION не найдена."
  fi
}

usage(){ echo "Использование: $0 {start|resume|attach|status|stop}"; }

cmd="${1:-usage}"
"$cmd"
