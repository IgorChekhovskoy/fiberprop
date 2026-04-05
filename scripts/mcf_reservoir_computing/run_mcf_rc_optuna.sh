#!/usr/bin/env bash
set -Eeuo pipefail

# === Автоопределение путей относительно текущего скрипта (минимальные правки) ===
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ==== Настройки (можно переопределить переменными окружения) ====
SESSION="${SESSION:-mcf_rc_20260222-145339}"                      # БАЗОВОЕ имя; при start будет выдано уникальное имя
SESSION_PREFIX="${SESSION_PREFIX:-mcf_rc_}"                      # префикс tmux-сессий расчёта
PROJECT_DIR="${PROJECT_DIR:-$REPO_ROOT}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
REQ_FILE="${REQ_FILE:-$PROJECT_DIR/requirements.txt}"
MAIN_PY="${MAIN_PY:-$SCRIPT_DIR/mcf_reservoir_computing.py}"
PY_ARGS="${PY_ARGS:-}"                            # сюда — аргументы твоему скрипту, если нужны
DB_PATH="${DB_PATH:-$SCRIPT_DIR/optuna_study.db}"   # ← тут: файл БД в подпапке подпроекта
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
JOURNAL_PATH="${JOURNAL_PATH:-$SCRIPT_DIR/mcf_optuna.journal}"  # уже в подпапке подпроекта

# Глушим внутренний трединг у BLAS/OMP/NumPy/Numba
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"

LOG_DIR="$SCRIPT_DIR/logs"; mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
DASH_LOG="$LOG_DIR/dashboard_$(date +%Y%m%d_%H%M%S).log"

# sqlite URL (если вернёшься к RDB)
if [[ "$DB_PATH" = /* ]]; then
  STORAGE_URL="sqlite:///$DB_PATH"   # даст sqlite:////abs/путь
else
  STORAGE_URL="sqlite:///$DB_PATH"
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# — файл с последним ИМЕНЕМ tmux-сессии этой обвязки (для attach/status/stop по умолчанию)
SESSION_FILE="$SCRIPT_DIR/.optuna_tmux_session"   # ← тут: в подпапке подпроекта

ensure_tmux(){ command -v tmux >/dev/null || { echo "tmux не установлен. sudo apt-get update && sudo apt-get install -y tmux"; exit 1; }; }
ensure_taskset(){ command -v taskset >/dev/null || { echo "taskset не найден (package util-linux). Установи: sudo apt-get install -y util-linux"; exit 1; }; }

_session_exists() {
  local ses="$1"
  [[ -n "$ses" ]] || return 1
  tmux has-session -t "=$ses" 2>/dev/null
}

_list_all_sessions() {
  tmux list-sessions -F '#S' 2>/dev/null || true
}

_list_matching_sessions() {
  _list_all_sessions | awk -v p="$SESSION_PREFIX" 'index($0, p) == 1'
}

_print_session_help() {
  if [[ -f "$SESSION_FILE" ]]; then
    echo "SESSION_FILE: $(< "$SESSION_FILE")"
  else
    echo "SESSION_FILE отсутствует."
  fi

  echo "Подходящие tmux-сессии с префиксом $SESSION_PREFIX:"
  _list_matching_sessions | sed 's/^/  /' || true

  echo "Все tmux-сессии:"
  _list_all_sessions | sed 's/^/  /' || true
}

_resolve_session_name() {
  local ses=""
  local matches=()
  local all=()

  # 1) Если файл есть и в нём имя живой сессии — берём его
  if [[ -f "$SESSION_FILE" ]]; then
    ses="$(< "$SESSION_FILE")"
    if _session_exists "$ses"; then
      echo "$ses"
      return 0
    fi
  fi

  # 2) Если среди расчётных tmux-сессий ровно одна — берём её
  mapfile -t matches < <(_list_matching_sessions)
  if (( ${#matches[@]} == 1 )); then
    echo "${matches[0]}"
    return 0
  fi

  # 3) Совсем последний фоллбэк: если у tmux вообще только одна сессия — берём её
  mapfile -t all < <(_list_all_sessions)
  if (( ${#all[@]} == 1 )); then
    echo "${all[0]}"
    return 0
  fi

  return 1
}

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
  fi

  # 4) Обязательные пакеты для работы скрипта
  echo "[bootstrap] Устанавливаю обязательные пакеты: optuna и optuna-dashboard"
  pip install optuna optuna-dashboard
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
    cpus="$(< /sys/fs/cgroup/cpuset.cpus)"
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

# --- функция генерации УНИКАЛЬНОГО имени tmux-сессии (минимальная интеграция) ---
_unique_session_name() {
  local base="$1"
  echo "${base}_$(date +%Y%m%d-%H%M%S)"
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
  WORKERS="${WORKERS:-$(($(physical_cores) - 20))}"
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

  # Всегда получить УНИКАЛЬНОЕ имя tmux-сессии
  local SESSION_ACTUAL
  SESSION_ACTUAL="$(_unique_session_name "$SESSION")"
  SESSION="$SESSION_ACTUAL"   # использовать далее в _spawn_*
  echo "$SESSION_ACTUAL" > "$SESSION_FILE"

  # Сгенерировать НОВОЕ имя study и запомнить его
  local STUDY_NAME_FILE="$SCRIPT_DIR/.study_name"
  local STUDY_NAME="${STUDY_NAME:-mcf_rc_$(date +%Y%m%d-%H%M%S)}"
  echo "$STUDY_NAME" > "$STUDY_NAME_FILE"

  _spawn_tmux_with_study "$STUDY_NAME"
}

# Новая короткая команда: продолжить предыдущую study
resume() {
  ensure_tmux
  ensure_taskset
  bootstrap

  local SES=""
  if SES="$(_resolve_session_name)"; then
    tmux attach -t "=$SES"
    exit 0
  fi

  # Если сессии нет — перезапуск по последнему study_name
  local STUDY_NAME_FILE="$SCRIPT_DIR/.study_name"   # ← тут: в подпапке подпроекта
  local STUDY_NAME="${STUDY_NAME:-}"
  if [[ -z "$STUDY_NAME" && -f "$STUDY_NAME_FILE" ]]; then
    STUDY_NAME="$(cat "$STUDY_NAME_FILE")"
  fi
  if [[ -z "$STUDY_NAME" ]]; then
    echo "Не найдено имя study. Либо задай STUDY_NAME=... $0 resume, либо сначала запусти $0 start"
    _print_session_help
    exit 1
  fi

  # Новое уникальное имя tmux-сессии на резюм
  SESSION="$(_unique_session_name "$SESSION")"
  echo "$SESSION" > "$SESSION_FILE"
  _spawn_tmux_with_study "$STUDY_NAME"
}

attach(){
  ensure_tmux
  local SES=""
  if ! SES="$(_resolve_session_name)"; then
    echo "Не удалось автоматически определить tmux-сессию для attach."
    _print_session_help
    echo "Укажи SESSION=... $0 attach"
    exit 1
  fi
  tmux attach -t "=$SES"
}

status(){
  local SES=""

  if tmux list-sessions >/dev/null 2>&1; then
    echo "Активные tmux-сессии:"
    tmux list-sessions -F '#S: #{session_windows} windows, attached=#{session_attached}'
    if [[ -f "$SESSION_FILE" ]]; then
      echo "Последняя, созданная этим скриптом: $(< "$SESSION_FILE")"
    fi
    if SES="$(_resolve_session_name)"; then
      echo "Сессия по умолчанию для attach/stop/resume: $SES"
    else
      echo "Сессия по умолчанию автоматически не определена."
    fi
    echo "Последние логи:"
    ls -1t "$LOG_DIR" | sed "s|^|$LOG_DIR/|"
  else
    echo "tmux-сервер не запущен (сессий нет)."
  fi
}

stop(){
  local SES=""
  if ! SES="$(_resolve_session_name)"; then
    echo "Не удалось автоматически определить tmux-сессию для остановки."
    _print_session_help
    echo "Укажи SESSION=... $0 stop"
    exit 1
  fi
  if tmux has-session -t "=$SES" 2>/dev/null; then
    echo "Останавливаю задачи (Ctrl-C) и закрываю сессию $SES…"
    for w in $(tmux list-windows -t "=$SES" -F '#W'); do
      tmux send-keys -t "=$SES:$w" C-c
    done
    sleep 1 || true
    tmux kill-session -t "=$SES" || true
    echo "✓ Остановлено."
  else
    echo "Сессия $SES не найдена."
  fi
}

usage(){ echo "Использование: $0 {start|resume|attach|status|stop}"; }

cmd="${1:-usage}"
"$cmd"