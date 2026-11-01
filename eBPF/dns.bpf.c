// dns.bpf.c -- cgroup/skb EGRESS: hvata DNS upite NA IZLASKU IZ PODA.
// Posto je program zakacen na cgroup konkretnog poda, identitet je uvek
// taj pod (ne CoreDNS). Parsira ime domena iz UDP/53 paketa.
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_endian.h>

#define ETH_HLEN 0            // cgroup_skb daje paket od IP headera (bez eth)
#define DNS_HDR  12
#define RAW_LEN  192
#define NAME_LEN 180

char LICENSE[] SEC("license") = "GPL";

struct event {
    __u64 cgroup_id;
    __u8  name[NAME_LEN];
    __u16 name_len;
    __u8  qtype_txt;         // 1 ako je TXT upit (zadrzano za kompatibilnost)
    __u16 qtype;            // pun QTYPE: 1=A, 28=AAAA, 16=TXT, 5=CNAME, 10=NULL, 15=MX...
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 20);
} events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct event);
} heap SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u8[RAW_LEN]);
} rawbuf SEC(".maps");

// Mapa odluka za KARANTIN: kljuc = cgroup_id poda, value = 1 (blokiraj).
// User space upisuje ovde kad detekcija oznaci pod. Kernel onda odbacuje.
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u64);
    __type(value, __u8);
} quarantine SEC(".maps");

// Dijagnostika: brojaci [0]=svi egress pozivi, [1]=UDP, [2]=DNS/53, [3]=poslato
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 4);
    __type(key, __u32);
    __type(value, __u64);
} dbg SEC(".maps");

static __always_inline void bump(__u32 i){
    __u64 *v = bpf_map_lookup_elem(&dbg, &i);
    if (v) __sync_fetch_and_add(v, 1);
}

SEC("cgroup_skb/egress")
int dns_egress(struct __sk_buff *skb)
{
    bump(0);                      // svaki egress poziv
    // Samo IPv4
    if (skb->family != 2)        // AF_INET = 2
        return 1;

    struct iphdr iph;
    if (bpf_skb_load_bytes(skb, 0, &iph, sizeof(iph)) < 0)
        return 1;
    if (iph.protocol != 17)             // UDP
        return 1;
    bump(1);                      // UDP

    __u32 ihl = iph.ihl * 4;
    struct udphdr udph;
    if (bpf_skb_load_bytes(skb, ihl, &udph, sizeof(udph)) < 0)
        return 1;
    if (udph.dest != bpf_htons(53))
        return 1;
    bump(2);                      // DNS

    __u64 cg = bpf_skb_cgroup_id(skb);

    // KARANTIN: ako je ovaj pod oznacen, odbaci DNS paket (0 = drop).
    __u8 *blocked = bpf_map_lookup_elem(&quarantine, &cg);
    if (blocked && *blocked)
        return 0;

    __u32 zero = 0;
    struct event *ev = bpf_map_lookup_elem(&heap, &zero);
    __u8 *raw = bpf_map_lookup_elem(&rawbuf, &zero);
    if (!ev || !raw)
        return 1;

    ev->cgroup_id = cg;
    __builtin_memset(ev->name, 0, sizeof(ev->name));
    ev->name_len = 0;
    ev->qtype_txt = 0;
    ev->qtype = 0;

    __u32 dns_off = ihl + sizeof(struct udphdr);      // pocetak DNS poruke
    __u32 qname_off = dns_off + DNS_HDR;               // pocetak imena

    __builtin_memset(raw, 0, RAW_LEN);

    // Jedan load_bytes, ali duzinu biramo prema stvarnoj velicini paketa.
    // DNS upiti su kratki; fiksni RAW_LEN=192 pukne ako je paket manji.
    __u32 avail = (skb->len > qname_off) ? (skb->len - qname_off) : 0;
    if (avail == 0)
        return 1;

    int ok = -1;
    if (avail >= RAW_LEN)
        ok = bpf_skb_load_bytes(skb, qname_off, raw, RAW_LEN);
    else if (avail >= 128)
        ok = bpf_skb_load_bytes(skb, qname_off, raw, 128);
    else if (avail >= 64)
        ok = bpf_skb_load_bytes(skb, qname_off, raw, 64);
    else if (avail >= 32)
        ok = bpf_skb_load_bytes(skb, qname_off, raw, 32);
    else
        ok = bpf_skb_load_bytes(skb, qname_off, raw, 16);
    if (ok < 0)
        return 1;

    __u32 i = 0, out = 0;
    int remain = 0;
    #pragma unroll
    for (int k = 0; k < NAME_LEN; k++) {
        if (i >= RAW_LEN - 1 || out >= NAME_LEN - 1)
            break;
        __u8 b = raw[i & (RAW_LEN - 1)];
        if (remain == 0) {
            if (b == 0) break;
            if (b > 63) break;
            if (out > 0) ev->name[out++] = '.';
            remain = b;
        } else {
            ev->name[out++] = b;
            remain--;
        }
        i++;
    }
    ev->name_len = out;

    // QTYPE je uvek poslednja 4 bajta pitanja: ... <ime> 00 QTYPE(2) QCLASS(2).
    // Citamo sa KRAJA paketa (skb->len - 4), pa radi i za duga imena koja
    // popune bafer (gde citanje posle imena ne bi stiglo do QTYPE-a).
    __u32 qlen = skb->len;
    if (qlen >= 4) {
        __u8 qt[2];
        if (bpf_skb_load_bytes(skb, qlen - 4, qt, 2) == 0) {
            ev->qtype = ((__u16)qt[0] << 8) | qt[1];
            if (ev->qtype == 16)
                ev->qtype_txt = 1;
        }
    }

    bump(3);                      // poslato u ringbuf
    bpf_ringbuf_output(&events, ev, sizeof(*ev), 0);
    return 1;                           // propusti paket (monitoring mod)
}
