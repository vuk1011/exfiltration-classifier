#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_endian.h>

#define DNS_HDR  12
#define RAW_LEN  192
#define NAME_LEN 180

char LICENSE[] SEC("license") = "GPL";

struct event {
    __u64 cgroup_id;
    __u8  name[NAME_LEN];
    __u16 name_len;
    __u8  qtype_txt;
    __u16 qtype;
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

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u64);
    __type(value, __u8);
} quarantine SEC(".maps");

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
    bump(0);
    if (skb->family != 2)
        return 1;

    struct iphdr iph;
    if (bpf_skb_load_bytes(skb, 0, &iph, sizeof(iph)) < 0)
        return 1;
    if (iph.protocol != 17)
        return 1;
    bump(1);

    __u32 ihl = iph.ihl * 4;
    struct udphdr udph;
    if (bpf_skb_load_bytes(skb, ihl, &udph, sizeof(udph)) < 0)
        return 1;
    if (udph.dest != bpf_htons(53))
        return 1;
    bump(2);

    __u64 cg = bpf_skb_cgroup_id(skb);
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

    __u32 dns_off = ihl + sizeof(struct udphdr);
    __u32 qname_off = dns_off + DNS_HDR;

    __builtin_memset(raw, 0, RAW_LEN);
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
    __u32 qlen = skb->len;
    if (qlen >= 4) {
        __u8 qt[2];
        if (bpf_skb_load_bytes(skb, qlen - 4, qt, 2) == 0) {
            ev->qtype = ((__u16)qt[0] << 8) | qt[1];
            if (ev->qtype == 16)
                ev->qtype_txt = 1;
        }
    }

    bump(3);
    bpf_ringbuf_output(&events, ev, sizeof(*ev), 0);
    return 1;
}
