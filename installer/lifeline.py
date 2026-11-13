"""End the tools of a run when the installer process itself is killed.

A launcher that runs the setup without a window (installer/cli.py --progress-lines) stops it by killing this process:
no Ctrl+Break and no SIGTERM arrive, so the cancel path of steps.py run_tool (kill_tree) never runs, and ninja, clang
or the shader translator would carry on for minutes with nobody waiting for them.

    arm()            once, at the start of the run, before any thread or tool starts
    watch(pid)       a tool started (steps.py run_tool); POSIX only, Windows needs no list
    unwatch(pid)     that tool ended

Windows: this process joins a job object that kills its processes when the last handle to the job closes. The only
handle is ours, so the tools die with us however we end; they join the job by being started from it.

POSIX: a forked watcher holds the read end of a pipe whose write end only this process has. The pipe closing means
this process is gone, and the watcher then kills the session of each tool still listed (kill_session; run_tool starts
every tool in a session of its own). A tool that ended is taken off the list first, so a clean exit kills nothing and
no recycled process id is ever hit.
"""
import ctypes
import os
import signal

_job = None    # Windows: the job handle, kept open for the life of the process
_pipe = None   # POSIX: the write end of the watcher's pipe

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectExtendedLimitInformation = 9


def _arm_windows():
    global _job
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                    ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                    ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                    ('SchedulingClass', wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ('ReadOperationCount', 'WriteOperationCount',
                                                         'OtherOperationCount', 'ReadTransferCount',
                                                         'WriteTransferCount', 'OtherTransferCount')]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', BasicLimits), ('IoInfo', IoCounters),
                    ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                    ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]

    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    k.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    k.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    job = k.CreateJobObjectW(None, None)   # not inheritable: no tool holds a second handle
    if not job:
        return False
    info = ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not (k.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info))
            and k.AssignProcessToJobObject(job, k.GetCurrentProcess())):
        k.CloseHandle(job)
        return False
    _job = job
    return True


def kill_session(session):
    """SIGKILL to every process of a session, for a tool started as a session leader (steps.py run_tool).

    The tool's process group is not enough: ninja gives each command it runs a process group of its own, so the
    compilers would finish their files after ninja is gone. They stay in the tool's session, which /proc lists. A few
    passes catch a process forked while one ran. Without /proc only the tool's group is killed."""
    try:
        os.killpg(session, signal.SIGKILL)
    except OSError:
        pass
    for _ in range(3):
        try:
            names = [n for n in os.listdir('/proc') if n.isdigit()]
        except OSError:
            return
        alive = False
        for name in names:
            try:
                with open(f'/proc/{name}/stat', encoding='utf-8', errors='replace') as f:
                    state, _parent, _group, sid = f.read().rsplit(')', 1)[1].split()[:4]
                if int(sid) == session and state != 'Z':
                    os.kill(int(name), signal.SIGKILL)
                    alive = True
            except (OSError, ValueError, IndexError):
                continue   # it ended while the list was read
        if not alive:
            return


def _watcher(read_end):
    """The watcher process: collect the listed tools (session leaders) until the pipe closes, then kill what is
    left of their sessions."""
    os.setsid()   # not in the launcher's process group: a signal sent to that group must not end the watcher first
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)   # the launcher reads our stdout to its end; the watcher must not keep it open
    groups, pending = set(), b''
    while True:
        try:
            chunk = os.read(read_end, 4096)
        except InterruptedError:
            continue
        if not chunk:
            break
        *lines, pending = (pending + chunk).split(b'\n')
        for line in lines:
            pid = int(line)
            (groups.add if pid > 0 else groups.discard)(abs(pid))
    for session in groups:
        kill_session(session)


def _arm_posix():
    global _pipe
    read_end, write_end = os.pipe()   # not inheritable: the tools never hold the write end
    if os.fork() == 0:
        try:
            os.close(write_end)
            _watcher(read_end)
        finally:
            os._exit(0)
    os.close(read_end)
    _pipe = write_end
    return True


def arm():
    """Tie the tools this process starts to its own life. Returns whether it worked; a run goes on either way."""
    if _job is not None or _pipe is not None:
        return True
    try:
        return _arm_windows() if os.name == 'nt' else _arm_posix()
    except (OSError, AttributeError):
        return False


def _tell(number):
    if _pipe is not None:
        try:
            os.write(_pipe, b'%d\n' % number)
        except OSError:
            pass


def watch(pid):
    _tell(pid)


def unwatch(pid):
    _tell(-pid)
