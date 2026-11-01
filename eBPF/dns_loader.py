import os
import sys
import math
import ctypes
import signal
from collections import Counter

ML_DIR = os.environ.get("ML_DIR", os.path.join(os.path.dirname(__file__), "..", "ml-classifier"))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ML_DIR, "model.pkl"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

classifier = None
try:
    sys.path.insert(0, ML_DIR)
    from classifier import DNSClassifier, DNSQuery  # noqa: E402
    if os.path.exists(MODEL_PATH):
        classifier = DNSClassifier.load(MODEL_PATH, threshold=THRESHOLD,
                                        feature_extractor=lambda q: extract_features(q.name.encode()))
        print(f"[ml] model loaded: {MODEL_PATH} (threshold={THRESHOLD})")
    else:
        print(f"[ml] WARNING: model not found at {MODEL_PATH}; using fallback threshold")
except Exception as ex:
    print(f"[ml] WARNING: ML unavailable ({ex}); using fallback threshold")

NAME_LEN = 180

class Event(ctypes.Structure):
    _fields_ = [
        ("cgroup_id", ctypes.c_uint64),
        ("name", ctypes.c_uint8 * NAME_LEN),
        ("name_len", ctypes.c_uint16),
        ("qtype_txt", ctypes.c_uint8),
        ("qtype", ctypes.c_uint16),
    ]

libbpf = ctypes.CDLL("libbpf.so")
libbpf.bpf_object__open_file.restype = ctypes.c_void_p
libbpf.bpf_object__open_file.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
libbpf.bpf_object__load.restype = ctypes.c_int
libbpf.bpf_object__load.argtypes = [ctypes.c_void_p]
libbpf.bpf_object__find_program_by_name.restype = ctypes.c_void_p
libbpf.bpf_object__find_program_by_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
libbpf.bpf_object__find_map_fd_by_name.restype = ctypes.c_int
libbpf.bpf_object__find_map_fd_by_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
libbpf.bpf_program__attach_cgroup.restype = ctypes.c_void_p
libbpf.bpf_program__attach_cgroup.argtypes = [ctypes.c_void_p, ctypes.c_int]
RINGBUF_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
libbpf.ring_buffer__new.restype = ctypes.c_void_p
libbpf.ring_buffer__new.argtypes = [ctypes.c_int, RINGBUF_CB, ctypes.c_void_p, ctypes.c_void_p]
libbpf.ring_buffer__poll.restype = ctypes.c_int
libbpf.ring_buffer__poll.argtypes = [ctypes.c_void_p, ctypes.c_int]
libbpf.ring_buffer__free.argtypes = [ctypes.c_void_p]
libbpf.bpf_object__close.argtypes = [ctypes.c_void_p]
libbpf.bpf_map_lookup_elem.restype = ctypes.c_int
libbpf.bpf_map_lookup_elem.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]

KUBEPODS = "/sys/fs/cgroup/kubepods"
pods = {}
links = []
stop_loop = False

import time
from collections import defaultdict, deque

FREQ_WINDOW = float(os.environ.get("FREQ_WINDOW", "30"))  
FREQ_THRESHOLD = int(os.environ.get("FREQ_THRESHOLD", "3"))  
freq_state = defaultdict(deque)

freq_alerted = {}

MITRE = {
    "ml":   "T1071.004 (Application Layer Protocol: DNS)",
    "txt":  "T1071.004 + T1048 (Exfiltration Over Alternative Protocol)",
}
QTYPE_NAMES = {1:"A", 28:"AAAA", 16:"TXT", 5:"CNAME", 10:"NULL",
               15:"MX", 2:"NS", 12:"PTR", 6:"SOA", 33:"SRV", 255:"ANY"}
def qtype_name(q):
    return QTYPE_NAMES.get(q, f"T{q}")

MITRE_SHORT = {
    "ml":   "T1071.004",
    "txt":  "T1071.004+T1048",
    "freq": "T1048.003",
}

DASHBOARD = os.environ.get("DASHBOARD", "0") == "1"
dash = {}
start_time = time.time()
total_events = 0

def dash_update(cgroup_id, ns, pod, score, name, ml_alarm=False, freq_alarm=False, mitre_tech=None, qtype=0):
    d = dash.get(cgroup_id)
    if d is None:
        d = {"ns": ns, "pod": pod, "queries": 0, "max_score": 0.0,
             "ml": 0, "freq": 0, "last": "", "status": "clean", "mitre": set(),
             "qtypes": Counter()}
        dash[cgroup_id] = d
    d["queries"] += 1
    d["max_score"] = max(d["max_score"], score)
    d["last"] = name[:40]
    if ml_alarm:
        d["ml"] += 1
    if freq_alarm:
        d["freq"] += 1
    if mitre_tech:
        d["mitre"].update(mitre_tech)
    d["qtypes"][qtype_name(qtype)] += 1
    if d["ml"] and d["freq"]:
        d["status"] = "HIGH"
    elif d["ml"] or d["freq"]:
        d["status"] = "suspicious"

def base_domain(name: str) -> str:
    labels = [l for l in name.split(".") if l]
    if len(labels) >= 2:
        return ".".join(labels[-2:])
    return name

def subdomain_part(name: str, base: str) -> str:
    if name.endswith(base):
        return name[: -len(base)].rstrip(".")
    return name

def freq_check(cgroup_id: int, name: str):
    now = time.time()
    base = base_domain(name)
    sub = subdomain_part(name, base)
    key = (cgroup_id, base)
    dq = freq_state[key]
    dq.append((now, sub))
    cutoff = now - FREQ_WINDOW
    while dq and dq[0][0] < cutoff:
        dq.popleft()
    unique = len({s for _, s in dq})
    triggered = unique >= FREQ_THRESHOLD
    return triggered, unique, base

def on_sigint(signum, frame):
    global stop_loop
    stop_loop = True

signal.signal(signal.SIGINT, on_sigint)

def calc_entropy_chars(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def extract_features(name_bytes: bytes) -> dict:
    name = name_bytes.decode("ascii", errors="ignore")
    labels = [l for l in name.split(".") if l != ""]
    joined = "".join(labels)
    length = len(joined)

    words = []
    cur = 0
    for ch in joined:
        if ch.isalpha():
            cur += 1
        else:
            if cur > 0:
                words.append(cur)
            cur = 0
    if cur > 0:
        words.append(cur)
    w_count = len(words)
    w_max = max(words) if words else 0

    digits = sum(1 for ch in joined if ch.isdigit())
    uppers = sum(1 for ch in joined if ch.isupper())

    return {
        "len": float(length),
        "subdomains_count": float(len(labels)),
        "w_count": float(w_count),
        "w_max": float(w_max),
        "entropy": calc_entropy_chars(joined),
        "w_max_ratio": (w_max / length) if length else 0.0,
        "w_count_ratio": (w_count / length) if length else 0.0,
        "digits_ratio": (digits / length) if length else 0.0,
        "uppercase_ratio": (uppers / length) if length else 0.0,
    }
def fallback_score(f: dict, is_txt: bool) -> float:
    s = 0.0
    if f["len"] > 40: s += 0.30
    if f["entropy"] > 3.7: s += 0.30
    if f["w_max"] > 25: s += 0.20
    if f["digits_ratio"] > 0.20: s += 0.10
    if is_txt: s += 0.10
    return min(1.0, s)

def score_name(name_bytes: bytes, is_txt: bool):
    f = extract_features(name_bytes)
    if classifier is not None:
        try:
            pred = classifier.classify(f)
            return pred.score, pred.is_attack, f
        except Exception as ex:
            print(f"  [ml] classification error ({ex}); using fallback")
    sc = fallback_score(f, is_txt)
    return sc, sc >= 0.70, f

def map_uid_to_pod(uid: str):
    log_pods = "/var/log/pods"
    if not os.path.exists(log_pods):
        return "unknown", "unknown"
    try:
        for entry in os.listdir(log_pods):
            if uid in entry:
                parts = entry.split("_")
                if len(parts) >= 2:
                    return parts[0], parts[1]
    except Exception:
        pass
    return "unknown", "unknown"

def attach_one(prog, path: str):
    try:
        cgfd = os.open(path, os.O_RDONLY)
    except OSError:
        return 0
    link = libbpf.bpf_program__attach_cgroup(prog, cgfd)
    os.close(cgfd)
    if not link:
        return 0
    links.append(link)
    return os.stat(path).st_ino

def attach_all(prog):
    for qos in ("burstable", "besteffort", "guaranteed"):
        base = os.path.join(KUBEPODS, qos)
        if not os.path.isdir(base):
            continue
        for d in os.listdir(base):
            if not d.startswith("pod"):
                continue
            pod_path = os.path.join(base, d)
            cg = attach_one(prog, pod_path)
            if cg:
                ns, pod_name = map_uid_to_pod(d[3:])
                pods[cg] = (ns, pod_name)
                print(f"[attach] {ns:<12} {pod_name:<40} cg={cg}")
                try:
                    for ce in os.listdir(pod_path):
                        if ce.startswith("."):
                            continue
                        cpath = os.path.join(pod_path, ce)
                        if os.path.isdir(cpath):
                            pods[os.stat(cpath).st_ino] = (ns, pod_name)
                except OSError:
                    pass

def handle_event(ctx, data, size):
    global total_events
    ev = Event.from_address(data)
    if not ev.name_len:
        return 0
    total_events += 1
    raw_name = bytes(ev.name[:ev.name_len])
    is_txt = bool(ev.qtype_txt)
    sc, is_attack, f = score_name(raw_name, is_txt)
    name_str = raw_name.decode("ascii", errors="ignore")

    ns, pod_name = pods.get(ev.cgroup_id, ("unknown", "unknown"))
    is_transit = (ns == "kube-system" and pod_name.startswith("coredns"))

    techniques = set()


    freq_alarm = False
    freq_hit, uniq, base_dom = freq_check(ev.cgroup_id, name_str)
    if freq_hit and not is_transit:
        key = (ev.cgroup_id, base_dom)
        last = freq_alerted.get(key, 0)
        if time.time() - last > FREQ_WINDOW / 2:
            freq_alerted[key] = time.time()
            freq_alarm = True
            techniques.add(MITRE_SHORT["freq"])
            if not DASHBOARD:
                print("\033[1;35m[FREQUENCY - suspicious pattern]\033[0m")
                print(f"  |- Namespace : {ns}")
                print(f"  |- Pod       : {pod_name}")
                print(f"  |- Domain    : {base_dom}")
                print(f"  |- {uniq} unique subdomains in < {FREQ_WINDOW:.0f}s "
                      f"(threshold {FREQ_THRESHOLD})")
                print(f"  \\- MITRE     : {MITRE['freq']}")
                print()

    ml_alarm = is_attack and not is_transit
    if ml_alarm:
        techniques.add(MITRE_SHORT["txt"] if is_txt else MITRE_SHORT["ml"])

    if not is_transit:
        dash_update(ev.cgroup_id, ns, pod_name, sc, name_str,
                    ml_alarm=ml_alarm, freq_alarm=freq_alarm, mitre_tech=techniques,
                    qtype=ev.qtype)

    if DASHBOARD:
        return 0  

    if is_attack:
        if is_transit:
            print(f"\033[1;33m[TRANSIT via CoreDNS]\033[0m {ns}/{pod_name}  {name_str} (score {sc:.2f})")
        else:
            mitre = MITRE["txt"] if is_txt else MITRE["ml"]
            print("\033[1;31m[ALERT]\033[0m")
            print(f"  |- Namespace  : {ns}")
            print(f"  |- Pod        : {pod_name}")
            print(f"  |- cgroup_id  : {ev.cgroup_id}")
            print(f"  |- DNS        : {name_str}")
            print(f"  |- TXT query  : {'yes' if is_txt else 'no'}")
            print(f"  |- Score      : {sc:.2f}")
            print(f"  |- Features   : entropy={f['entropy']:.2f} len={f['len']:.0f} "
                  f"w_max={f['w_max']:.0f} subdomains={f['subdomains_count']:.0f} "
                  f"digits={f['digits_ratio']:.2f}")
            print(f"  \\- MITRE      : {mitre}")
            print()
    else:
        print(f"{ns:<12} {pod_name:<30} score={sc:.2f} len={f['len']:.0f} {name_str}")
    return 0


def render_dashboard():
    from rich.table import Table
    from rich.console import Console
    _ = Console()
    t = Table(title="DNS Exfiltration Monitor",
              show_lines=True, expand=False)
    t.add_column("Pod / details", no_wrap=True)
    t.add_column("Queries", justify="right", width=8)
    t.add_column("Score", justify="right", width=6)
    t.add_column("Status", no_wrap=True, width=12)
    t.add_column("MITRE ATT&CK", no_wrap=True, width=18)

    rank = {"HIGH": 0, "suspicious": 1, "clean": 2}
    rows = sorted(dash.values(), key=lambda d: (rank.get(d["status"], 3), -d["max_score"]))
    for d in rows:
        status = d["status"]
        mitre = ", ".join(sorted(d["mitre"])) if d["mitre"] else "-"
        if status == "HIGH":
            st = "[bold red]HIGH RISK[/]"
            podcol = f"[bold red]{d['ns']}/{d['pod']}[/]"
            mcol = f"[bold red]{mitre}[/]"
        elif status == "suspicious":
            st = "[yellow]suspicious[/]"
            podcol = f"[yellow]{d['ns']}/{d['pod']}[/]"
            mcol = f"[yellow]{mitre}[/]"
        else:
            st = "[green]clean[/]"
            podcol = f"{d['ns']}/{d['pod']}"
            mcol = f"[dim]{mitre}[/]"

        t.add_row(podcol, str(d["queries"]), f"{d['max_score']:.2f}", st, mcol)
        details = f"   \\ ML:{d['ml']}  Freq:{d['freq']}  last: {d['last']}"
        t.add_row(f"[dim]{details}[/]", "", "", "", "")
        qd = "  ".join(f"{k}:{v}" for k, v in d["qtypes"].most_common())
        t.add_row(f"[dim]   \\ types: {qd}[/]", "", "", "", "")
    elapsed = int(time.time() - start_time)
    t.caption = (f"Queries: {total_events}  ·  Pods: {len(dash)}  ·  "
                 f"Uptime: {elapsed}s  ·  MITRE: T1071.004 / T1048  ·  Ctrl-C to quit")
    return t

def main():
    bpf_obj_file = "dns.bpf.o"
    if not os.path.exists(bpf_obj_file):
        print(f"Error: {bpf_obj_file} not found. Run 'make'.")
        sys.exit(1)
    obj = libbpf.bpf_object__open_file(bpf_obj_file.encode(), None)
    if not obj:
        print("Error: bpf_object__open_file failed"); sys.exit(1)
    if libbpf.bpf_object__load(obj) != 0:
        print("Error: bpf_object__load failed"); sys.exit(1)
    prog = libbpf.bpf_object__find_program_by_name(obj, b"dns_egress")
    if not prog:
        print("Error: program 'dns_egress' not found"); sys.exit(1)

    attach_all(prog)
    if not pods:
        print("WARNING: no pod cgroup was attached", file=sys.stderr)

    events_fd = libbpf.bpf_object__find_map_fd_by_name(obj, b"events")
    cb = RINGBUF_CB(handle_event)
    rb = libbpf.ring_buffer__new(events_fd, cb, None, None)
    if not rb:
        print("Error: ring_buffer creation failed"); sys.exit(1)

    if DASHBOARD:
        try:
            from rich.live import Live
        except ImportError:
            print("Error: dashboard needs 'rich' (sudo python3 -m pip install "
                  "--break-system-packages rich). Or run without DASHBOARD=1.")
            sys.exit(1)
        try:
            with Live(render_dashboard(), refresh_per_second=4, screen=True) as live:
                while not stop_loop:
                    err = libbpf.ring_buffer__poll(rb, 100)
                    if err < 0 and err != -4:
                        break
                    live.update(render_dashboard())
        finally:
            libbpf.ring_buffer__free(rb)
            libbpf.bpf_object__close(obj)
            print_summary()
    else:
        print("Monitoring DNS egress from pods (Python + ML)... Ctrl-C to quit")
        print("-" * 64)
        try:
            while not stop_loop:
                err = libbpf.ring_buffer__poll(rb, 100)
                if err < 0 and err != -4:
                    break
        finally:
            libbpf.ring_buffer__free(rb)
            libbpf.bpf_object__close(obj)
            print_summary()


def print_summary():
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Total DNS queries : {total_events}")
    print(f"  Pods monitored    : {len(dash)}")
    suspicious = [d for d in dash.values() if d["status"] != "clean"]
    print(f"  Suspicious pods   : {len(suspicious)}")
    for d in sorted(suspicious, key=lambda x: -x["max_score"]):
        mitre = ", ".join(sorted(d["mitre"])) if d["mitre"] else "-"
        qd = " ".join(f"{k}:{v}" for k, v in d["qtypes"].most_common())
        print(f"    - {d['ns']}/{d['pod']}: status={d['status']} "
              f"max_score={d['max_score']:.2f} ML_alarms={d['ml']} freq_alarms={d['freq']} "
              f"MITRE=[{mitre}]")
        print(f"        query types: {qd}")
    print("  MITRE ATT&CK      : T1071.004 (DNS C2), T1048 (Exfiltration)")
    print("=" * 60)

if __name__ == "__main__":
    main()
