#!/usr/bin/env bash
set -Eeuo pipefail

# === Автоопределение путей относительно текущего скрипта (минимальные правки) ===
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ==== Настройки (можно переопределить переменными окружения) ====
SESSION="${SESSION:-mcf_rc}"
PROJECT_DIR="${PROJECT_DIR:-$REPO_ROOT}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
REQ_FILE="${REQ_FILE:-$PROJECT_DIR/requirements.txt}"
MAIN_PY="${MAIN_PY:-$SCRIPT_DIR/mcf_reservoir_computing.py}"
PY_ARGS="${PY_ARGS:-}"                         # сюда — аргументы твоему скрипту, если нужны
DB_PATH="${DB_PATH:-$PROJECT_DIR/optuna_study.db}"   # оставлено на будущее (RDB)
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"                           # (как у тебя на линуксе)
JOURNAL_PATH="${JOURNAL_PATH:-$SCRIPT_DIR/mcf_optuna.journal}"

# Глушим внутренний трединг у BLAS/OMP/NumPy/Numba
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"

LOG_DIR="$PROJECT_DIR/logs"; mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
DASH_LOG="$LOG_DIR/dashboard_$(date +%Y%m%d_%H%M%S).log"

# sqlite URL (если вернёшься к RDB)
if [[ "$DB_PATH" = /* ]]; then
  STORAGE_URL="sqlite:///$DB_PATH"   # даст sqlite:////abs/путь
else
  STORAGE_URL="sqlite:///$DB_PATH"
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

ensure_tmux(){ command -v tmux >/dev/null || { echo "tmux не установлен. sudo apt-get update && sudo apt-get install -y tmux"; exit 1; }; }
ensure_taskset(){ command -v taskset >/dev/null || { echo "taskset не найден (package util-linux). Установи: sudo apt-get install -y util-linux"; exit 1; }; }

bootstrap() {
  echo "[bootstrap] Проект: $PROJECT_DIR"
  mkdir -p "$PROJECT_DIR"
  cd "$PROJECT_DIR"

  # 1) venv (создать, если нет)
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[bootstrap] Создаю venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
  # 2) Активировать и обновить pip
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip >/dev/null
  # 3) requirements.txt (если есть)
  if [[ -f "$REQ_FILE" ]]; then
    echo "[bootstrap] Устанавливаю зависимости из $REQ_FILE"
    pip install -r "$REQ_FILE"
  else
    echo "[bootstrap] Внимание: $REQ_FILE не найден — пропускаю установку зависимостей."
    echo "           Убедись, что optuna и optuna-dashboard стоят, либо добавь их в requirements.txt."
  fi
}

# --- физические ядра (сокеты × cores per socket); фоллбэки: /proc/cpuinfo → nproc ---
physical_cores() {
  if command -v lscpu >/dev/null; then
    local sockets cores
    sockets="$(lscpu | awk -F: '/^Socket\(s\)/{gsub(/ /,"",$2); print $2}')"
    cores="$(lscpu | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2); print $2}')"
    if [[ -n "$sockets" && -n "$cores" && "$sockets" -gt 0 && "$cores" -gt 0 ]]; then
      echo $(( sockets * cores ))
      return
    fi
  fi
  if [[ -r /proc/cpuinfo ]]; then
    awk '
      /physical id/ {p=$4}
      /core id/     {c=$4; print p "-" c}
    ' /proc/cpuinfo | sort -u | wc -l
    return
  fi
  nproc --all
}

# --- разворачиваем строку формата "0-3,5,7-9" в список чисел по одному в строке
expand_cpulist() {
  local s="$1"
  awk -v RS= -v list="$s" '
    BEGIN{
      n=split(list,a,",");
      for(i=1;i<=n;i++){
        if(a[i] ~ /^[0-9]+-[0-9]+$/){
          split(a[i],b,"-");
          for(j=b[1]; j<=b[2]; j++) print j;
        } else if(a[i] ~ /^[0-9]+$/){
          print a[i];
        }
      }
    }'
}

# --- список ДОСТУПНЫХ логических CPU: cpuset(cgroup) → sysfs(online) → lscpu → nproc --all
logical_cpu_list() {
  local cpus=""
  # cgroup v2 (обычно): /sys/fs/cgroup/cpuset.cpus
  if [[ -r /sys/fs/cgroup/cpuset.cpus ]]; then
    cpus="$(< /sys/fs/cgroup/cpuset.cpus)"
    [[ -n "$cpus" ]] && { expand_cpulist "$cpus"; return; }
  fi
  # cgroup v1: /sys/fs/cgroup/cpuset/cpuset.cpus
  if [[ -r /sys/fs/cgroup/cpuset/cpuset.cpus ]]; then
    cpus="$(< /sys/fs/cgroup/cpuset/cpus)"
    [[ -n "$cpus" ]] && { expand_cpulist "$cpus"; return; }
  fi
  # sysfs список «онлайн» CPU, напр. "0-7,16-23"
  if [[ -r /sys/devices/system/cpu/online ]]; then
    cpus="$(< /sys/devices/system/cpu/online)"
    [[ -n "$cpus" ]] && { expand_cpulist "$cpus"; return; }
  fi
  # lscpu -п=CPU,ONLINE (CSV, строки с ONLINE=1/Y)
  if command -v lscpu >/dev/null; then
    lscpu -p=CPU,ONLINE 2>/dev/null | awk -F',' '/^[^#]/{ if($2==1 || $2=="Y") print $1 }'
    return
  fi
  # фоллбэк — всё, что установлено (0..nproc-1)
  local n="$(( $(nproc --all 2>/dev/null || echo 1) - 1 ))"
  seq 0 "${n#-1}"
}

# --- общая функция запуска одной сессии (reuse в start/resume) ---
_spawn_tmux_with_study() {
  local study_name="$1"
  export MCF_STUDY_NAME="$study_name"
  echo "[start] STUDY_NAME = $MCF_STUDY_NAME" | tee -a "$RUN_LOG"

  mkdir -p "$(dirname "$DB_PATH")"
  mkdir -p "$(dirname "$JOURNAL_PATH")"
  touch "$JOURNAL_PATH"   # создаём пустой журнал заранее (для дашборда)

  # Кол-во воркеров = физическим ядрам (можно переопределить WORKERS в env)
  WORKERS="${WORKERS:-$(physical_cores)}"
  if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || (( WORKERS < 1 )); then WORKERS=1; fi
  echo "[start] WORKERS = $WORKERS (физические ядра)"

  # Получаем список логических CPU и пинем воркеры по одному CPU на процесс
  mapfile -t LOG_CPU < <(logical_cpu_list | awk '/^[0-9]+$/')
  if (( ${#LOG_CPU[@]} == 0 )); then LOG_CPU=(0); fi
  echo "[start] Доступные логические CPU: ${LOG_CPU[*]}"

  # Окна run1..runN: каждый — отдельный ПРОЦЕСС Python, пинут к одному логическому CPU
  for ((i=0; i<WORKERS; i++)); do
    win="run$((i+1))"
    if (( i == 0 )); then
      tmux new-session -d -s "$SESSION" -n "$win"
    else
      tmux new-window -t "$SESSION" -n "$win"
    fi
    tmux send-keys -t "$SESSION:$win" "cd '$PROJECT_DIR'" C-m
    tmux send-keys -t "$SESSION:$win" "source '$VENV_DIR/bin/activate' || true" C-m

    # безопасное сопоставление воркера к CPU (по кругу, если WORKERS > логических CPU)
    cpu="${LOG_CPU[$(( i % ${#LOG_CPU[@]} ))]}"

    tmux send-keys -t "$SESSION:$win" "echo \"[run$((i+1))] CPU set: $cpu, OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS\" | tee -a '$RUN_LOG'" C-m

    # ВАЖНО: MCF_BASH=1 → Python поставит n_jobs=1; taskset -c принимает одиночный CPU/список/диапазон.
    tmux send-keys -t "$SESSION:$win" "MCF_BASH=1 taskset -c $cpu python '$MAIN_PY' $PY_ARGS 2>&1 | tee -a '$RUN_LOG'" C-m
  done

  # Отдельное окно: optuna-dashboard (JournalStorage)
  tmux new-window -t "$SESSION" -n dash
  tmux send-keys -t "$SESSION:dash" "cd '$PROJECT_DIR'" C-m
  tmux send-keys -t "$SESSION:dash" "source '$VENV_DIR/bin/activate' || true" C-m
  tmux send-keys -t "$SESSION:dash" "echo \"[dashboard] JOURNAL: $JOURNAL_PATH\" | tee -a '$DASH_LOG'" C-m
  tmux send-keys -t "$SESSION:dash" "optuna-dashboard --storage-class JournalFileStorage '$JOURNAL_PATH' --host $HOST --port $PORT 2>&1 | tee -a '$DASH_LOG'" C-m

  echo
  echo "✓ Запущено в tmux-сессии: $SESSION"
  echo "  Логи: $RUN_LOG  и  $DASH_LOG"
  echo "  Optuna Dashboard:  http://$HOST:$PORT/"
  echo
  echo "=== Подсказки ==="
  echo "• WORKERS=$WORKERS процессов; каждый закреплён за ОДНИМ логическим CPU (taskset)."
  echo "• В Python коде включи load_if_exists=True и study_name=$MCF_STUDY_NAME — тогда будет резюмирование."
  echo
}

start() {
  ensure_tmux
  ensure_taskset
  bootstrap

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Сессия $SESSION уже запущена. Используй: $0 attach"
    exit 0
  fi

  # Сгенерировать НОВОЕ имя study и запомнить его
  local STUDY_NAME_FILE="$PROJECT_DIR/.study_name"
  local STUDY_NAME="${STUDY_NAME:-mcf_rc_$(date +%Y%m%d-%H%M%S)}"
  echo "$STUDY_NAME" > "$STUDY_NAME_FILE"

  _spawn_tmux_with_study "$STUDY_NAME"
}

# Новая короткая команда: продолжить предыдущую study
resume() {
  ensure_tmux
  ensure_taskset
  bootstrap

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Сессия $SESSION уже запущена. Используй: $0 attach"
    exit 0
  fi

  local STUDY_NAME_FILE="$PROJECT_DIR/.study_name"
  local STUDY_NAME="${STUDY_NAME:-}"
  if [[ -z "$STUDY_NAME" ]]; then
    if [[ -f "$STUDY_NAME_FILE" ]]; then
      STUDY_NAME="$(cat "$STUDY_NAME_FILE")"
    fi
  fi
  if [[ -z "$STUDY_NAME" ]]; then
    echo "Не найдено имя study. Либо задай STUDY_NAME=... $0 resume, либо сначала запусти $0 start"
    exit 1
  fi

  # НЕ перезаписываем .study_name — продолжаем ровно ту, что указана
  _spawn_tmux_with_study "$STUDY_NAME"
}

attach(){ ensure_tmux; tmux attach -t "$SESSION"; }

status(){
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Сессия $SESSION активна. Окна:"
    tmux list-windows -t "$SESSION"
    echo "Дашборд: http://$HOST:$PORT/"
    ls -1t "$LOG_DIR" | sed "s|^|log: $LOG_DIR/|"
    if [[ -f "$PROJECT_DIR/.study_name" ]]; then
      echo "Последнее сохранённое study_name: $(cat "$PROJECT_DIR/.study_name")"
    fi
  else
    echo "Сессия $SESSION не найдена."
    if [[ -f "$PROJECT_DIR/.study_name" ]]; then
      echo "Последнее сохранённое study_name: $(cat "$PROJECT_DIR/.study_name")"
    fi
  fi
}

stop(){
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Останавливаю задачи (Ctrl-C) и закрываю сессию $SESSION…"
    for w in $(tmux list-windows -t "$SESSION" -F '#W'); do
      tmux send-keys -t "$SESSION:$w" C-c
    done
    sleep 1 || true
    tmux kill-session -t "$SESSION" || true
    echo "✓ Остановлено."
  else:
    echo "Сессия $SESSION не найдена."
  fi
}

usage(){ echo "Использование: $0 {start|resume|attach|status|stop}"; }

cmd="${1:-usage}"
"$cmd"
