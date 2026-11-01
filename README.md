# eBPF DNS Tunneling Detection & Automated Pod Quarantine

A kernel-level security monitoring and automated containment solution for Kubernetes environments built on **eBPF (CO-RE)**. 

The system intercepts outbound DNS traffic from individual Kubernetes pods before it hits CoreDNS, performs heuristic and entropy analysis to detect DNS tunneling techniques (e.g., APT34 / Helix Kitten C2 channels), and enforces immediate network-level isolation on compromised workloads.

---

## Architecture Overview

### 1. In-Kernel Probe (`dns.bpf.c`)
- **Hook point:** Attached to each pod's control group as a `cgroup_skb/egress` program.
- **Filtering:** Filters for IPv4 packets targeting UDP destination port 53.
- **Inspection:** Inspects the raw packet buffer to extract domain names (converting DNS label-length encoding into dot notation) and detects query types (specifically tracking high-capacity TXT records).
- **Early-Drop Mitigation:** Checks the BPF hash map `quarantine`. If the packet's `cgroup_id` is flagged, the packet is instantly dropped (`return 0`) directly inside the kernel network path.
- **Telemetry:** Streams metadata events to userspace asynchronously using a high-throughput BPF ring buffer.

### 2. Userspace Agent (`dns_loader.py`)
- **Cgroup Discovery:** Scans `/sys/fs/cgroup/kubepods/` QoS trees (`burstable`, `besteffort`, `guaranteed`) and attaches the BPF program to each active pod and container cgroup.
- **Workload Identification:** Maps the cgroup's filesystem directory inode number (`cgroup_id`) to the specific Kubernetes namespace and pod name by cross-referencing `/var/log/pods`.
- **Heuristic & Feature Extraction:**
  - **Shannon Entropy:** Measures domain randomness ($H = -\sum p \log_2 p$).
  - **Query Length & Label Length:** Evaluates domain length ($> 40$) and maximum subdomain length ($> 25$).
  - **Digit Ratio:** Measures numerical character density ($> 0.20$).
  - **Query Type:** Adds weight to TXT requests frequently abused for payload transport.
- **CoreDNS Disambiguation:** Filters transit traffic traversing `kube-system/coredns` to identify the originating pod instead of the intermediate resolver.

---

## Prerequisites

- **Linux Kernel:** Version 5.8+ with BTF support (`/sys/kernel/btf/vmlinux`) and unified cgroup v2 hierarchy.
- **Host Tools:** `clang`, `llvm`, `libbpf-dev`, `bpftool`, `make`.
- **Python Environment:** Python 3.8+ with standard library (`ctypes`, `subprocess`, `math`).
- **Kubernetes Engine:** Cluster access with administrative credentials (tested on `k0s` / Vanilla Kubernetes).

---

## Deployment & Setup

### 1. Deploy Kubernetes Infrastructure
Apply the application workloads, Ingress rules, MetalLB configuration, and defensive NetworkPolicies:

```bash
sudo k0s kubectl apply -f metallb-config.yaml
sudo k0s kubectl apply -f ingress.yaml
sudo k0s kubectl apply -f k8s.yaml
```

Verify that all pods in the `investment-fund` namespace are running:
```bash
sudo k0s kubectl -n investment-fund get pods
```

### 2. Compile the eBPF Kernel Object
Generate the target architecture's `vmlinux.h` and compile `dns.bpf.c` into an ELF object file:

```bash
make clean
make
```
### 3. Run the Detector

Start the Python-based monitor with root privileges (required for cgroup attachment and BPF syscalls):

```bash
sudo DASHBOARD=1 ML_DIR=../ml-classifier MODEL_PATH=../ml-classifier/model.pkl python3 dns_loader.py
```

Upon startup, the loader identifies existing pod cgroups, binds the program, and starts polling the ring buffer:
```text
[attach] investment-fund employee-corrupt-service-59bc7d559-6h8xk  cg=4026532890
[attach] investment-fund employee-78df6f8849-p2xkl                cg=4026532895
[attach] investment-fund director-6cf9575f78-2d7lm                cg=4026532901
----------------------------------------------------------------
```

---

## Threat Simulation & Verification

The manifest includes `employee-corrupt-service`, an insider-threat/compromised service configured to generate automated DNS tunneling traffic destined for `hacker.internal`:

```yaml
data:
  DNS_TRAFFIC_ENABLED: "true"
  DNS_ZONE: "hacker.internal"
  DNS_INTERVAL_SECONDS: "5"
```
