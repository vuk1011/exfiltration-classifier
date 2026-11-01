#!/usr/bin/env python3
import os
import sys
import math
import glob
import ctypes
import signal
import subprocess
from collections import Counter

# --- CTYPES DEFINICIJE ZA STRUKTURE I LIBBPF ---
NAME_LEN = 180

class Event(ctypes.Structure):
    _fields_ = [
        ("cgroup_id", ctypes.c_uint64),
        ("name", ctypes.c_uint8 * NAME_LEN),
        ("name_len", ctypes.c_uint16),
        ("qtype_txt", ctypes.c_uint8),
    ]

# Učitavanje libbpf deljene biblioteke
libbpf = ctypes.CDLL("libbpf.so")

# Tipovi funkcija iz libbpf
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

# Callback tip za ring_buffer: int (*ring_buffer_sample_fn)(void *ctx, void *data, size_t size)
RINGBUF_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
libbpf.ring_buffer__new.restype = ctypes.c_void_p
libbpf.ring_buffer__new.argtypes = [ctypes.c_int, RINGBUF_CB, ctypes.c_void_p, ctypes.c_void_p]

libbpf.ring_buffer__poll.restype = ctypes.c_int
libbpf.ring_buffer__poll.argtypes = [ctypes.c_void_p, ctypes.c_int]

libbpf.ring_buffer__free.argtypes = [ctypes.c_void_p]
libbpf.bpf_object__close.argtypes = [ctypes.c_void_p]

# Sistemski bpf poziv za bpf_map_update_elem / lookup
libbpf.bpf_map_lookup_elem.restype = ctypes.c_int
libbpf.bpf_map_lookup_elem.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]

libbpf.bpf_map_update_elem.restype = ctypes.c_int
libbpf.bpf_map_update_elem.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]

# --- GLOBALNO STANJE ---
KUBEPODS = "/sys/fs/cgroup/kubepods"
pods = {}          # cgroup_id -> (namespace, pod)
links = []         # Reference na bpf_link objekte da ih GC ne bi obrisao
quarantined = set()
quarantine_fd = -1
stop_loop = False

def on_sigint(signum, frame):
    global stop_loop
    stop_loop = True

signal.signal(signal.SIGINT, on_sigint)

# --- DETEKCIJA I STATISTIKA ---
def calc_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

def extract_features(name_bytes: bytes):
    name_str = name_bytes.decode('ascii', errors='ignore')
    entropy = calc_entropy(name_bytes)
    length = len(name_bytes)
    
    digits = sum(1 for c in name_str if c.isdigit())
    digit_ratio = (digits / length) if length > 0 else 0.0
    
    labels = name_str.split('.')
    max_label_len = max((len(lbl) for lbl in labels), default=0)
    
    return {
        "length": length,
        "entropy": entropy,
        "digit_ratio": digit_ratio,
        "max_label_len": max_label_len,
        "name": name_str
    }

def calculate_score(f: dict, is_txt: bool) -> float:
    s = 0.0
    if f["length"] > 40:
        s += 0.30
    if f["entropy"] > 3.7:
        s += 0.30
    if f["max_label_len"] > 25:
        s += 0.20
    if f["digit_ratio"] > 0.20:
        s += 0.10
    if is_txt:
        s += 0.10
    return min(1.0, s)

# --- POD MAPIRANJE I KARANTIN ---
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

def quarantine_label(ns: str, pod: str):
    key = f"{ns}/{pod}"
    if key in quarantined:
        return
    quarantined.add(key)
    
    cmd = ["sudo", "k0s", "kubectl", "-n", ns, "label", "pod", pod, "is_suspicious=true", "--overwrite"]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0:
            print(f"  \033[1;33m>> KARANTIN: {ns}/{pod} egress blokiran (is_suspicious=true)\033[0m")
        else:
            print(f"  >> karantin nije uspeo za {ns}/{pod} (rc={res.returncode})")
    except Exception as ex:
        print(f"  >> Greska pri pokretanju kubectl: {ex}")

def quarantine_pod_bpf(cgroup_id: int):
    if quarantine_fd >= 0:
        cg_key = ctypes.c_uint64(cgroup_id)
        val = ctypes.c_uint8(1)
        libbpf.bpf_map_update_elem(quarantine_fd, ctypes.byref(cg_key), ctypes.byref(val), 0)

# --- CGROUP ATTACH ---
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
    st = os.stat(path)
    return st.st_ino

def attach_all(prog):
    qos_dirs = ["burstable", "besteffort", "guaranteed"]
    for qos in qos_dirs:
        base = os.path.join(KUBEPODS, qos)
        if not os.path.isdir(base):
            continue
        
        for d in os.listdir(base):
            if not d.startswith("pod"):
                continue
            
            pod_path = os.path.join(base, d)
            cg = attach_one(prog, pod_path)
            if cg:
                uid = d[3:]
                ns, pod_name = map_uid_to_pod(uid)
                pods[cg] = (ns, pod_name)
                print(f"[attach] {ns:<12} {pod_name:<40} cg={cg}")
                
                # Kontejnerski cgroup-ovi unutar pod foldera
                try:
                    for ce in os.listdir(pod_path):
                        if ce.startswith('.'):
                            continue
                        cpath = os.path.join(pod_path, ce)
                        if os.path.isdir(cpath):
                            cst = os.stat(cpath)
                            pods[cst.st_ino] = (ns, pod_name)
                except OSError:
                    pass

# --- EVENT OBRADA (RING BUFFER CALLBACK) ---
def handle_event(ctx, data, size):
    ev = Event.from_address(data)
    if not ev.name_len:
        return 0
    
    raw_name = bytes(ev.name[:ev.name_len])
    is_txt = bool(ev.qtype_txt)
    f = extract_features(raw_name)
    sc = calculate_score(f, is_txt)
    
    ns, pod_name = pods.get(ev.cgroup_id, ("nepoznat", "nepoznat"))
    is_transit = (ns == "kube-system" and pod_name.startswith("coredns"))
    
    if sc >= 0.70:
        if is_transit:
            print(f"\033[1;33m[TRANZIT kroz CoreDNS]\033[0m {ns}/{pod_name}  {f['name']} (skor {sc:.2f})")
        else:
            print("\033[1;31m[ALARM - APT34 / HELIX KITTEN]\033[0m")
            print(f"  ├─ Namespace : {ns}")
            print(f"  ├─ Pod       : {pod_name}")
            print(f"  ├─ cgroup_id : {ev.cgroup_id}")
            print(f"  ├─ DNS       : {f['name']}")
            print(f"  ├─ TXT upit  : {'da' if is_txt else 'ne'}")
            print(f"  ├─ Skor      : {sc:.2f}/1.00")
            print(f"  └─ Atributi  : entropija={f['entropy']:.2f} duzina={f['length']} maxlabel={f['max_label_len']} cifre={f['digit_ratio']:.2f}")
            
            # Karantin preko k0s kubectl labele
            if pod_name != "nepoznat":
                quarantine_label(ns, pod_name)
                # quarantine_pod_bpf(ev.cgroup_id) # Ako zelis i cist eBPF drop
            print()
    else:
        print(f"{ns:<12} {pod_name:<30} e={f['entropy']:.2f} len={f['length']} {f['name']}")
        
    return 0

# --- MAIN ---
def main():
    global quarantine_fd
    
    bpf_obj_file = "dns.bpf.o"
    if not os.path.exists(bpf_obj_file):
        print(f"Greska: {bpf_obj_file} nije pronadjen. Pokreni 'make' da generises .o fajl.")
        sys.exit(1)
        
    obj = libbpf.bpf_object__open_file(bpf_obj_file.encode('utf-8'), None)
    if not obj:
        print("Greska: bpf_object__open_file failed")
        sys.exit(1)
        
    if libbpf.bpf_object__load(obj) != 0:
        print("Greska: bpf_object__load failed")
        sys.exit(1)
        
    prog = libbpf.bpf_object__find_program_by_name(obj, b"dns_egress")
    if not prog:
        print("Greska: Program 'dns_egress' nije pronadjen")
        sys.exit(1)
        
    quarantine_fd = libbpf.bpf_object__find_map_fd_by_name(obj, b"quarantine")
    
    attach_all(prog)
    if len(pods) == 0:
        print("UPOZORENJE: nijedan pod cgroup nije zakacen", file=sys.stderr)
        
    events_fd = libbpf.bpf_object__find_map_fd_by_name(obj, b"events")
    cb = RINGBUF_CB(handle_event)
    rb = libbpf.ring_buffer__new(events_fd, cb, None, None)
    if not rb:
        print("Greska: Kreiranje ring_buffer-a nije uspelo")
        sys.exit(1)
        
    print("Pratim DNS na izlasku iz podova (Python)... Ctrl-C za kraj")
    print("-" * 64)
    
    try:
        while not stop_loop:
            err = libbpf.ring_buffer__poll(rb, 100)
            if err < 0 and err != -4: # -4 je EINTR (od signala)
                break
    finally:
        print("\nKraj.")
        dbg_fd = libbpf.bpf_object__find_map_fd_by_name(obj, b"dbg")
        if dbg_fd >= 0:
            lbl = ["svi egress", "UDP", "DNS/53", "poslato"]
            for k in range(4):
                key = ctypes.c_uint32(k)
                val = ctypes.c_uint64(0)
                libbpf.bpf_map_lookup_elem(dbg_fd, ctypes.byref(key), ctypes.byref(val))
                print(f"  dbg[{k}] {lbl[k]:<12} = {val.value}")
                
        libbpf.ring_buffer__free(rb)
        libbpf.bpf_object__close(obj)

if __name__ == "__main__":
    main()
