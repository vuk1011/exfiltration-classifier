CLANG ?= clang
ARCH := $(shell uname -m | sed 's/x86_64/x86/')

.PHONY: all clean run

all: vmlinux.h dns.bpf.o
vmlinux.h:
	bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h

dns.bpf.o: dns.bpf.c vmlinux.h
	$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) \
		-c dns.bpf.c -o dns.bpf.o

run: all
	sudo ML_DIR=../ml-classifier MODEL_PATH=../ml-classifier/model.pkl python3 dns_loader.py

clean:
	rm -f vmlinux.h dns.bpf.o dns_loader
