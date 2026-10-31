# Makefile za CO-RE eBPF DNS monitor.
# make        -> generise vmlinux.h, kompajlira eBPF i loader
# sudo ./dns_loader
CLANG ?= clang
ARCH := $(shell uname -m | sed 's/x86_64/x86/')

.PHONY: all clean run

all: vmlinux.h dns.bpf.o dns_loader

# 1) vmlinux.h iz BTF-a (ne treba kernel source, samo bpftool + BTF)
vmlinux.h:
	bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h

# 2) eBPF program -> bajtkod .o
dns.bpf.o: dns.bpf.c vmlinux.h
	$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) \
		-c dns.bpf.c -o dns.bpf.o

# 3) loader (user space)
dns_loader: dns_loader.c
	$(CLANG) -g -O2 dns_loader.c -lbpf -lelf -lz -lm -o dns_loader

run: all
	sudo ./dns_loader

clean:
	rm -f vmlinux.h dns.bpf.o dns_loader
