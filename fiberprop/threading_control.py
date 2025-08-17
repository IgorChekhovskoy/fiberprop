# fiberprop/threading_control.py
from __future__ import annotations
from typing import Optional, Union
from contextlib import contextmanager
import os, sys

# ── утилиты ────────────────────────────────────────────────────────────────

def _resolve(n: Union[int, str, None]) -> Optional[int]:
    """None/'default'/'inherit' -> None; 'max'/'all'/'*' -> os.cpu_count(); int -> max(1,int(n))."""
    if n is None:
        return None
    s = str(n).strip().lower()
    if s in ("default", "inherit"):
        return None
    if s in ("max", "all", "*"):
        return os.cpu_count() or 1
    return max(1, int(n))

def _set_blas_openmp_threads(n: Optional[int]) -> None:
    """Лимит для MKL/BLAS/OpenMP. Предпочтение — через threadpoolctl, иначе env."""
    try:
        from threadpoolctl import ThreadpoolController
        ThreadpoolController().limit(limits=n, user_api=("blas", "openmp"))
        return
    except Exception:
        pass
    if n is None:
        return
    s = str(n)
    # MKL имеет приоритет над OMP_NUM_THREADS. :contentReference[oaicite:1]{index=1}
    os.environ["MKL_NUM_THREADS"] = s
    os.environ["OMP_NUM_THREADS"] = s
    os.environ["OPENBLAS_NUM_THREADS"] = s
    os.environ["VECLIB_MAXIMUM_THREADS"] = s
    os.environ["NUMEXPR_NUM_THREADS"] = s

def _set_torch_threads(n: Optional[int]) -> None:
    """Настройка потоков PyTorch (CPU)."""
    try:
        import torch
    except Exception:
        return
    try:
        if n is not None:
            # перекрывает env; вызывать до вычислений. :contentReference[oaicite:2]{index=2}
            torch.set_num_threads(int(n))
        if n is not None:
            # можно только один раз до старта inter-op; игнорируем ошибку. :contentReference[oaicite:3]{index=3}
            try:
                torch.set_num_interop_threads(int(n))
            except Exception:
                pass
    except Exception:
        pass

# ── публичный API ─────────────────────────────────────────────────────────

def configure_threads(num_threads: Union[int, str, None] = "default",
                      *, wait_policy_active: bool = True,
                      kmp_blocktime_infinite: bool = True) -> None:
    """
    Глобально выставляет число потоков для NumPy/BLAS/OpenMP, PyTorch и (корректно) Numba.
    - num_threads: None/'default' — как есть; 'max' — все ядра; int — конкретное число ≥1.
    - wait_policy_active=True  ⇒ OMP_WAIT_POLICY=ACTIVE (минимальная латентность).
    - kmp_blocktime_infinite=True ⇒ KMP_BLOCKTIME=infinite.
    """
    n = _resolve(num_threads)

    # Политика ожидания (ACTIVE + infinite). :contentReference[oaicite:4]{index=4}
    if wait_policy_active:
        os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
    if kmp_blocktime_infinite:
        os.environ["KMP_BLOCKTIME"] = "infinite"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

    # BLAS/OpenMP (MKL/OMP/…)
    _set_blas_openmp_threads(n)

    # Numba: НЕ менять NUMBA_NUM_THREADS после импорта!
    numba_loaded = "numba" in sys.modules
    if not numba_loaded:
        # Настраиваем слой до импорта
        os.environ["NUMBA_THREADING_LAYER"] = "omp"
        if n is not None:
            os.environ["NUMBA_NUM_THREADS"] = str(int(n))  # потолок процесса
    else:
        # Рантайм уже поднят — меняем только фактический лимит
        try:
            import numba
            if n is not None:
                numba.set_num_threads(int(n))
        except Exception:
            pass

    # PyTorch
    _set_torch_threads(n)

def threading_report() -> dict:
    """Сводка по потокам (BLAS/OpenMP и PyTorch)."""
    rep = {
        "os_cpu_count": os.cpu_count(),
        "env": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS","OMP_WAIT_POLICY","KMP_BLOCKTIME","KMP_AFFINITY",
            "NUMBA_THREADING_LAYER","NUMBA_NUM_THREADS"
        )},
        "numpy_blas_openmp": [],
        "torch": {},
        "numba": {}
    }
    try:
        from threadpoolctl import threadpool_info
        rep["numpy_blas_openmp"] = threadpool_info()
    except Exception as ex:
        rep["numpy_blas_openmp"] = [{"info_error": repr(ex)}]
    try:
        import torch
        rep["torch"]["intra_op"] = torch.get_num_threads()
        try:
            rep["torch"]["inter_op"] = torch.get_num_interop_threads()
        except Exception as ex:
            rep["torch"]["inter_op_error"] = repr(ex)
    except Exception as ex:
        rep["torch"]["error"] = repr(ex)
    try:
        import numba
        rep["numba"]["threads"] = numba.get_num_threads()
    except Exception:
        pass
    return rep

@contextmanager
def temporary_thread_limits(num_threads: Union[int, str, None]):
    """
    Временная переустановка лимитов (например, внутри Optuna trial).
    Корректно меняет Numba через API и откатывает назад.
    """
    # сохранить окружение (кроме NUMBA_* — их нельзя дёргать после импорта)
    prev_env = {k: os.environ.get(k) for k in (
        "OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS","OMP_WAIT_POLICY","KMP_BLOCKTIME","KMP_AFFINITY"
    )}

    # сохранить текущие лимиты для отката
    prev_numba = None
    try:
        import numba
        prev_numba = numba.get_num_threads()
    except Exception:
        pass

    prev_torch_intra = None
    prev_torch_inter = None
    try:
        import torch
        prev_torch_intra = torch.get_num_threads()
        try:
            prev_torch_inter = torch.get_num_interop_threads()
        except Exception:
            pass
    except Exception:
        pass

    # применить новые лимиты
    configure_threads(num_threads)

    try:
        yield
    finally:
        # откат env
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        # откат BLAS/OpenMP через threadpoolctl, если доступен
        try:
            from threadpoolctl import ThreadpoolController
            ThreadpoolController().restore_original_limits()
        except Exception:
            pass
        # откат Numba через API
        if prev_numba is not None:
            try:
                import numba
                numba.set_num_threads(int(prev_numba))
            except Exception:
                pass
        # откат PyTorch
        try:
            import torch
            if prev_torch_intra is not None:
                torch.set_num_threads(int(prev_torch_intra))
            # inter-op мог быть «одноразово» установлен — откатываем, если можно
            if prev_torch_inter is not None:
                try:
                    torch.set_num_interop_threads(int(prev_torch_inter))
                except Exception:
                    pass
        except Exception:
            pass

def physical_cpu_count() -> int:
    """
    Возвращает число ФИЗИЧЕСКИХ ядер (не логических).
    Порядок попыток:
      1) psutil.cpu_count(logical=False)
      2) Windows: wmic / PowerShell CIM (NumberOfCores)
      3) macOS: sysctl -n hw.physicalcpu
      4) Linux: lscpu (Core(s) per socket * Socket(s)) или парсинг /proc/cpuinfo
      5) Fallback: os.cpu_count()  (логические) — как «последний шанс»
    """
    import os, sys, subprocess, shlex

    # 1) psutil (кроссплатформенно; возвращает физические ядра или None)
    try:
        import psutil  # psutil.readthedocs.io: cpu_count(logical=False) → физические ядра
        n = psutil.cpu_count(logical=False)
        if isinstance(n, int) and n > 0:
            return n
    except Exception:
        pass

    plat = sys.platform

    # 2) Windows — NumberOfCores (WMIC → PowerShell CIM)
    if plat.startswith("win"):
        # wmic (есть даже на старых системах; может быть помечен deprecated, но работает)
        try:
            out = subprocess.check_output(["wmic", "cpu", "get", "NumberOfCores"], text=True, stderr=subprocess.DEVNULL)
            nums = [int(s) for s in out.split() if s.isdigit()]
            if nums:
                return sum(nums)
        except Exception:
            pass
        # PowerShell (новее и надёжнее): суммируем NumberOfCores по всем сокетам
        try:
            ps_cmd = "Get-CimInstance Win32_Processor | Select-Object -Expand NumberOfCores | Measure-Object -Sum | " \
                     "Select-Object -Expand Sum"
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps_cmd],
                                          text=True, stderr=subprocess.DEVNULL)
            n = int(out.strip())
            if n > 0:
                return n
        except Exception:
            pass

    # 3) macOS — sysctl hw.physicalcpu
    if plat == "darwin":
        try:
            out = subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.physicalcpu"],
                                          text=True, stderr=subprocess.DEVNULL)
            n = int(out.strip())
            if n > 0:
                return n
        except Exception:
            pass

    # 4) Linux — lscpu (Core(s) per socket × Socket(s)), иначе /proc/cpuinfo
    if plat.startswith("linux"):
        # lscpu
        try:
            out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
            cores_per_socket, sockets = None, None
            for line in out.splitlines():
                if "Core(s) per socket:" in line:
                    cores_per_socket = int(line.split(":")[1].strip())
                elif "Socket(s):" in line:
                    sockets = int(line.split(":")[1].strip())
            if cores_per_socket and sockets:
                n = cores_per_socket * sockets
                if n > 0:
                    return n
        except Exception:
            pass
        # /proc/cpuinfo: считаем уникальные пары (physical id, core id)
        try:
            phys_ids = set()
            cur_phys, cur_core = None, None
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    if ln.startswith("physical id"):
                        cur_phys = ln.split(":")[1].strip()
                    elif ln.startswith("core id"):
                        cur_core = ln.split(":")[1].strip()
                    elif not ln.strip():  # конец записи CPU
                        if cur_phys is not None and cur_core is not None:
                            phys_ids.add((cur_phys, cur_core))
                        cur_phys, cur_core = None, None
                # финальная запись без пустой строки
                if cur_phys is not None and cur_core is not None:
                    phys_ids.add((cur_phys, cur_core))
            if phys_ids:
                return len(phys_ids)
        except Exception:
            pass

    # 5) Fallback — логические (лучше, чем 0)
    try:
        n = os.cpu_count() or 1
        return int(n)
    except Exception:
        return 1
