# Makefile za CO-RE eBPF DNS monitor (MONITORING + ML preko Pythona).
# make        -> generise vmlinux.h i dns.bpf.o
# sudo python3 dns_loader.py   (ML verzija)
CLANG ?= clang
ARCH := $(shell uname -m | sed 's/x86_64/x86/')

.PHONY: all clean run

# Gradimo samo eBPF objekat; loader je Python (dns_loader.py) sa ML-om.
all: vmlinux.h dns.bpf.o

# vmlinux.h iz BTF-a (ne treba kernel source, samo bpftool + BTF)
vmlinux.h:
	bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h

# eBPF program -> bajtkod .o
dns.bpf.o: dns.bpf.c vmlinux.h
	$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) \
		-c dns.bpf.c -o dns.bpf.o

run: all
	sudo ML_DIR=../ml-classifier MODEL_PATH=../ml-classifier/model.pkl python3 dns_loader.py

clean:
	rm -f vmlinux.h dns.bpf.o dns_loader
