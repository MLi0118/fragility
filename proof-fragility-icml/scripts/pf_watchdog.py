"""pf_watchdog.py <pid> <minutes>: kill the process tree of <pid> if the whole
tree uses < 5 s of CPU over <minutes> (a silent vLLM hang). Exit 0 when <pid>
exits on its own, 2 after killing it. Lean REPL workers call setsid, so the
tree is walked by parent pid, not by session."""
import os
import sys
import time
import signal

HZ = os.sysconf("SC_CLK_TCK")


def procs():
    out = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                f = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()
                out[int(p)] = (int(f[1]), (int(f[11]) + int(f[12])) / HZ)  # ppid, utime+stime
            except (OSError, IndexError, ValueError):
                pass
    return out


def tree(root):
    """{pid: cpu seconds} for root and all its descendants."""
    ps, kids, seen, stack = procs(), {}, {}, [root]
    for pid, (ppid, _) in ps.items():
        kids.setdefault(ppid, []).append(pid)
    while stack:
        p = stack.pop()
        if p in ps:
            seen[p] = ps[p][1]
            stack += kids.get(p, [])
    return seen


root, window = int(sys.argv[1]), int(sys.argv[2])
# CPU used per minute, summed per process: a tree total would drop when a vLLM
# engine exits between stages and read as idle.
prev, used = tree(root), []
while os.path.exists(f"/proc/{root}"):
    time.sleep(60)
    cur = tree(root)
    used = (used + [sum(max(0.0, c - prev.get(p, 0.0)) for p, c in cur.items())])[-window:]
    prev = cur
    if len(used) == window and sum(used) < 5:
        print(f"watchdog: tree of {root} used {sum(used):.1f}s CPU in {window} min; "
              f"killing {len(cur)} procs", flush=True)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for p in tree(root):
                try:
                    os.kill(p, sig)
                except OSError:
                    pass
            time.sleep(15)
        sys.exit(2)
