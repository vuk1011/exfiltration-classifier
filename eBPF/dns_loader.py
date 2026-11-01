#!/usr/bin/env python3
# dns_loader.py -- cgroup/skb egress monitoring + ML klasifikator.
# SAMO MONITORING: karantin (dodavanje labele is_suspicious=true) je uklonjen.
import os
import sys
import math
import ctypes
import signal
from collections import Counter

# --- ML klasifikator (ml-classifier modul) ---
# Dodaj putanju do ml-classifier paketa i ucitaj model.
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
        print(f"[ml] model ucitan: {MODEL_PATH} (threshold={THRESHOLD})")
    else:
        print(f"[ml] UPOZORENJE: model nije nadjen na {MODEL_PATH}; koristim rezervni prag")
except Exception as ex:
    print(f"[ml] UPOZORENJE: ML nedostupan ({ex}); koristim rezervni prag")

# --- CTYPES DEFINICIJE ZA STRUKTURE I LIBBPF ---
NAME_LEN = 180

class Event(ctypes.Structure):
    _fields_ = [
        ("cgroup_id", ctypes.c_uint64),
        ("name", ctypes.c_uint8 * NAME_LEN),
        ("name_len", ctypes.c_uint16),
        ("qtype_txt", ctypes.c_uint8),
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

# --- GLOBALNO STANJE ---
KUBEPODS = "/sys/fs/cgroup/kubepods"
pods = {}
links = []
stop_loop = False

def on_sigint(signum, frame):
    global stop_loop
    stop_loop = True

signal.signal(signal.SIGINT, on_sigint)

# --- FEATURE EKSTRAKCIJA (9 feature-a koje model ocekuje) ---
# Napomena: definicije prate semantiku Mendeley DNS-exfil dataseta.
#   len               : broj karaktera imena (bez tacaka)
#   subdomains_count  : broj labela (poddomena), tj. delova izmedju tacaka
#   w_count           : broj "reci" (maksimalnih nizova slova a-z/A-Z)
#   w_max             : duzina najduze reci
#   entropy           : Shannon entropija nad karakterima imena (bez tacaka)
#   w_max_ratio       : w_max / len
#   w_count_ratio     : w_count / len
#   digits_ratio      : udeo cifara u imenu (bez tacaka)
#   uppercase_ratio   : udeo velikih slova u imenu (bez tacaka)
def calc_entropy_chars(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def extract_features(name_bytes: bytes) -> dict:
    name = name_bytes.decode("ascii", errors="ignore")
    labels = [l for l in name.split(".") if l != ""]
    joined = "".join(labels)                 # ime bez tacaka
    length = len(joined)

    # reci = maksimalni nizovi slova
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

# Rezervni prag (ako ML nije dostupan): grubo, na osnovu par feature-a.
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
            print(f"  [ml] greska u klasifikaciji ({ex}); rezervni prag")
    sc = fallback_score(f, is_txt)
    return sc, sc >= 0.70, f

# --- POD MAPIRANJE ---
def map_uid_to_pod(uid: str):
    log_pods = "/var/log/pods"
    if not os.path.exists(log_pods):
        return "nepoznat", "nepoznat"
    try:
        for entry in os.listdir(log_pods):
            if uid in entry:
                parts = entry.split("_")
                if len(parts) >= 2:
                    return parts[0], parts[1]
    except Exception:
        pass
    return "nepoznat", "nepoznat"

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

# --- EVENT OBRADA (SAMO MONITORING, BEZ KARANTINA) ---
def handle_event(ctx, data, size):
    ev = Event.from_address(data)
    if not ev.name_len:
        return 0
    raw_name = bytes(ev.name[:ev.name_len])
    is_txt = bool(ev.qtype_txt)
    sc, is_attack, f = score_name(raw_name, is_txt)
    name_str = raw_name.decode("ascii", errors="ignore")

    ns, pod_name = pods.get(ev.cgroup_id, ("nepoznat", "nepoznat"))
    is_transit = (ns == "kube-system" and pod_name.startswith("coredns"))

    if is_attack:
        if is_transit:
            print(f"\033[1;33m[TRANZIT kroz CoreDNS]\033[0m {ns}/{pod_name}  {name_str} (skor {sc:.2f})")
        else:
            print("\033[1;31m[ALARM - APT34 / HELIX KITTEN]\033[0m")
            print(f"  ├─ Namespace : {ns}")
            print(f"  ├─ Pod       : {pod_name}")
            print(f"  ├─ cgroup_id : {ev.cgroup_id}")
            print(f"  ├─ DNS       : {name_str}")
            print(f"  ├─ TXT upit  : {'da' if is_txt else 'ne'}")
            print(f"  ├─ Skor      : {sc:.2f}")
            print(f"  └─ Atributi  : entropija={f['entropy']:.2f} len={f['len']:.0f} "
                  f"w_max={f['w_max']:.0f} subdomains={f['subdomains_count']:.0f} "
                  f"cifre={f['digits_ratio']:.2f}")
            print()
    else:
        print(f"{ns:<12} {pod_name:<30} score={sc:.2f} len={f['len']:.0f} {name_str}")
    return 0

def main():
    bpf_obj_file = "dns.bpf.o"
    if not os.path.exists(bpf_obj_file):
        print(f"Greska: {bpf_obj_file} nije pronadjen. Pokreni 'make'.")
        sys.exit(1)
    obj = libbpf.bpf_object__open_file(bpf_obj_file.encode(), None)
    if not obj:
        print("Greska: bpf_object__open_file failed"); sys.exit(1)
    if libbpf.bpf_object__load(obj) != 0:
        print("Greska: bpf_object__load failed"); sys.exit(1)
    prog = libbpf.bpf_object__find_program_by_name(obj, b"dns_egress")
    if not prog:
        print("Greska: Program 'dns_egress' nije pronadjen"); sys.exit(1)

    attach_all(prog)
    if not pods:
        print("UPOZORENJE: nijedan pod cgroup nije zakacen", file=sys.stderr)

    events_fd = libbpf.bpf_object__find_map_fd_by_name(obj, b"events")
    cb = RINGBUF_CB(handle_event)
    rb = libbpf.ring_buffer__new(events_fd, cb, None, None)
    if not rb:
        print("Greska: Kreiranje ring_buffer-a nije uspelo"); sys.exit(1)

    print("Monitoring DNS na izlasku iz podova (Python + ML)... Ctrl-C za kraj")
    print("-" * 64)
    try:
        while not stop_loop:
            err = libbpf.ring_buffer__poll(rb, 100)
            if err < 0 and err != -4:
                break
    finally:
        print("\nKraj.")
        libbpf.ring_buffer__free(rb)
        libbpf.bpf_object__close(obj)

if __name__ == "__main__":
    main()
