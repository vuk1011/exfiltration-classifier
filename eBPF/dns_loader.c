// dns_loader.c -- kaci cgroup_skb/egress na SVAKI pod cgroup, mapira
// cgroup_id -> namespace/pod, detektuje APT34, i moze da stavi pod u karantin.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <math.h>
#include <ctype.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

#define KUBEPODS "/sys/fs/cgroup/kubepods"
#define MAX_PODS 512

struct event {
    unsigned long long cgroup_id;
    unsigned char name[180];
    unsigned short name_len;
    unsigned char qtype_txt;
};

// cgroup_id -> (namespace, pod)
struct pod_entry {
    unsigned long long cgroup_id;
    char ns[64];
    char pod[128];
};
static struct pod_entry pods[MAX_PODS];
static int npods = 0;

static int quarantine_fd = -1;

// ---------- feature-i (ime) ----------
struct dns_features { double entropy; int length; double digit_ratio; int max_label_len; };

static double calc_entropy(const unsigned char *s, size_t n) {
    if (!n) return 0.0;
    int c[256] = {0};
    for (size_t i=0;i<n;i++) c[s[i]]++;
    double e=0.0;
    for (int i=0;i<256;i++) if (c[i]) { double p=(double)c[i]/n; e -= p*(log(p)/log(2.0)); }
    return e;
}
static void extract(const struct event *e, struct dns_features *f) {
    f->length=e->name_len; f->entropy=calc_entropy(e->name,e->name_len);
    int dig=0,cur=0; f->max_label_len=0;
    for (int i=0;i<e->name_len;i++){
        if (isdigit(e->name[i])) dig++;
        if (e->name[i]=='.'){ if(cur>f->max_label_len)f->max_label_len=cur; cur=0; }
        else cur++;
    }
    if (cur>f->max_label_len)f->max_label_len=cur;
    f->digit_ratio = e->name_len? (double)dig/e->name_len : 0.0;
}
// privremeni "ML" prag (zamenjuje se stablom kasnije)
static double score(const struct dns_features *f, int txt) {
    double s=0;
    if (f->length>40) s+=0.30;
    if (f->entropy>3.7) s+=0.30;
    if (f->max_label_len>25) s+=0.20;
    if (f->digit_ratio>0.20) s+=0.10;
    if (txt) s+=0.10;                 // TXT = APT34 omiljeni tip
    return s>1.0?1.0:s;
}

// ---------- mapiranje UID -> namespace/pod preko /var/log/pods ----------
// /var/log/pods ima foldere oblika <ns>_<pod>_<uid>
static void map_uid_to_pod(const char *uid_nodash_or_dash, char *ns, char *pod) {
    strcpy(ns,"nepoznat"); strcpy(pod,"nepoznat");
    DIR *d=opendir("/var/log/pods");
    if(!d) return;
    struct dirent *e;
    while((e=readdir(d))){
        if(strstr(e->d_name, uid_nodash_or_dash)){
            char tmp[256]; strncpy(tmp,e->d_name,sizeof(tmp)-1); tmp[sizeof(tmp)-1]=0;
            char *a=strtok(tmp,"_"); char *b=strtok(NULL,"_");
            if(a&&b){ strncpy(ns,a,63); ns[63]=0; strncpy(pod,b,127); pod[127]=0; }
            break;
        }
    }
    closedir(d);
}

// Karantin preko Kubernetes labele: postavi is_suspicious=true na pod.
// NetworkPolicy "allow-egress-if-clean" tada vise ne vazi za njega -> egress staje.
// Pamtimo koje smo vec karantinirali, da ne zovemo kubectl u petlji.
static char quarantined[256][192];
static int nq = 0;
static int already_quarantined(const char *ns, const char *pod){
    char key[192]; snprintf(key,sizeof(key),"%s/%s",ns,pod);
    for(int i=0;i<nq;i++) if(strcmp(quarantined[i],key)==0) return 1;
    if(nq<256){ strncpy(quarantined[nq],key,191); nq++; }
    return 0;
}
static void quarantine_label(const char *ns, const char *pod){
    char cmd[512];
    snprintf(cmd,sizeof(cmd),
        "sudo k0s kubectl -n %s label pod %s is_suspicious=true --overwrite >/dev/null 2>&1",
        ns, pod);
    int rc = system(cmd);
    if(rc==0) printf("  \033[1;33m>> KARANTIN: %s/%s egress blokiran (is_suspicious=true)\033[0m\n", ns, pod);
    else      printf("  >> karantin nije uspeo za %s/%s (rc=%d)\n", ns, pod, rc);
}

static const char* pod_name(unsigned long long cg, const char **ns_out){
    for(int i=0;i<npods;i++) if(pods[i].cgroup_id==cg){ *ns_out=pods[i].ns; return pods[i].pod; }
    *ns_out="nepoznat"; return "nepoznat";
}

// zakaci program na jedan cgroup dir; vrati cgroup_id (inode) ili 0
static unsigned long long attach_one(struct bpf_program *prog, const char *path){
    int cgfd=open(path,O_RDONLY);
    if(cgfd<0) return 0;
    struct bpf_link *l=bpf_program__attach_cgroup(prog,cgfd);
    close(cgfd);
    if(!l) return 0;
    struct stat st; if(stat(path,&st)<0) return 0;
    return (unsigned long long)st.st_ino;    // cgroup_id == inode
}

// prodji /sys/fs/cgroup/kubepods/<qos>/pod<UID> i zakaci na svaki
static void attach_all(struct bpf_program *prog){
    const char *qos[]={"burstable","besteffort","guaranteed"};
    for(int q=0;q<3;q++){
        char base[256]; snprintf(base,sizeof(base),"%s/%s",KUBEPODS,qos[q]);
        DIR *d=opendir(base); if(!d) continue;
        struct dirent *e;
        while((e=readdir(d))){
            if(strncmp(e->d_name,"pod",3)!=0) continue;
            char path[512]; snprintf(path,sizeof(path),"%s/%s",base,e->d_name);
            unsigned long long cg=attach_one(prog,path);
            if(cg && npods<MAX_PODS){
                // UID je deo posle "pod"; probaj i sa i bez crtica
                const char *uid=e->d_name+3;
                char ns[64],pod[128];
                map_uid_to_pod(uid,ns,pod);
                pods[npods].cgroup_id=cg;
                strncpy(pods[npods].ns,ns,63);
                strncpy(pods[npods].pod,pod,127);
                npods++;
                printf("[attach] %-12s %-40s cg=%llu\n", ns, pod, cg);

                // Paket dolazi iz KONTEJNERSKOG cgroup-a (jedan nivo dublje),
                // cija je cgroup_id razlicita od pod-level. Mapiraj i njih
                // na isti pod/namespace, da bi prepoznavanje radilo.
                DIR *pd=opendir(path);
                if(pd){
                    struct dirent *ce;
                    while((ce=readdir(pd)) && npods<MAX_PODS){
                        if(ce->d_name[0]=='.') continue;
                        char cpath[600];
                        snprintf(cpath,sizeof(cpath),"%s/%s",path,ce->d_name);
                        struct stat cst;
                        if(stat(cpath,&cst)<0 || !S_ISDIR(cst.st_mode)) continue;
                        pods[npods].cgroup_id=(unsigned long long)cst.st_ino;
                        strncpy(pods[npods].ns,ns,63);
                        strncpy(pods[npods].pod,pod,127);
                        npods++;
                    }
                    closedir(pd);
                }
            }
        }
        closedir(d);
    }
}

// ---------- karantin ----------
static void quarantine_pod(unsigned long long cg){
    unsigned char one=1;
    if(quarantine_fd>=0) bpf_map_update_elem(quarantine_fd,&cg,&one,0);
}

static volatile int stop; static void on_sigint(int s){stop=1;}

static int handle(void *ctx,void *data,size_t len){
    struct event *e=data;
    if(!e->name_len) return 0;
    struct dns_features f; extract(e,&f);
    double sc=score(&f,e->qtype_txt);

    const char *ns; const char *pod=pod_name(e->cgroup_id,&ns);

    // Da li je ovo CoreDNS tranzit (prosledjen upit), a ne izvorni pod?
    int is_transit = (strcmp(ns,"kube-system")==0 && strncmp(pod,"coredns",7)==0);

    if(sc>=0.70){
        if(is_transit){
            // Tranzit kroz CoreDNS: isti upit, ali nije izvor napada.
            printf("\033[1;33m[TRANZIT kroz CoreDNS]\033[0m %s/%s  %s (skor %.2f)\n",
                   ns, pod, e->name, sc);
        } else {
            // Izvorni pod: pun alarm + karantin.
            printf("\033[1;31m[ALARM - APT34 / HELIX KITTEN]\033[0m\n");
            printf("  ├─ Namespace : %s\n", ns);
            printf("  ├─ Pod       : %s\n", pod);
            printf("  ├─ cgroup_id : %llu\n", e->cgroup_id);
            printf("  ├─ DNS       : %s\n", e->name);
            printf("  ├─ TXT upit  : %s\n", e->qtype_txt?"da":"ne");
            printf("  ├─ Skor      : %.2f/1.00\n", sc);
            printf("  └─ Atributi  : entropija=%.2f duzina=%d maxlabel=%d cifre=%.2f\n",
                   f.entropy,f.length,f.max_label_len,f.digit_ratio);
            // KARANTIN: promeni labelu poda -> NetworkPolicy mu sece egress.
            // (samo jednom po podu, da ne spamuje kubectl)
            if(strcmp(pod,"nepoznat")!=0 && !already_quarantined(ns,pod)){
                quarantine_label(ns, pod);
            }
            printf("\n");
        }
    } else {
        printf("%-12s %-30s e=%.2f len=%d %s\n", ns, pod, f.entropy, f.length, e->name);
    }
    return 0;
}

int main(void){
    struct bpf_object *obj=bpf_object__open_file("dns.bpf.o",NULL);
    if(!obj){fprintf(stderr,"open failed\n");return 1;}
    if(bpf_object__load(obj)){fprintf(stderr,"load failed\n");return 1;}

    struct bpf_program *prog=bpf_object__find_program_by_name(obj,"dns_egress");
    if(!prog){fprintf(stderr,"prog not found\n");return 1;}

    quarantine_fd=bpf_object__find_map_fd_by_name(obj,"quarantine");

    attach_all(prog);
    if(npods==0) fprintf(stderr,"UPOZORENJE: nijedan pod cgroup nije zakacen\n");

    int map_fd=bpf_object__find_map_fd_by_name(obj,"events");
    struct ring_buffer *rb=ring_buffer__new(map_fd,handle,NULL,NULL);
    if(!rb){fprintf(stderr,"ringbuf failed\n");return 1;}

    signal(SIGINT,on_sigint);
    printf("Pratim DNS na izlasku iz podova... Ctrl-C za kraj\n");
    printf("----------------------------------------------------------------\n");
    while(!stop){ int err=ring_buffer__poll(rb,100); if(err<0&&err!=-4) break; }
    printf("\nKraj.\n");

    // Ispis dijagnostickih brojaca
    int dbg_fd = bpf_object__find_map_fd_by_name(obj, "dbg");
    if (dbg_fd >= 0) {
        const char *lbl[4] = {"svi egress", "UDP", "DNS/53", "poslato"};
        for (unsigned int k = 0; k < 4; k++) {
            unsigned long long v = 0;
            bpf_map_lookup_elem(dbg_fd, &k, &v);
            printf("  dbg[%u] %-12s = %llu\n", k, lbl[k], v);
        }
    }

    ring_buffer__free(rb); bpf_object__close(obj);
    return 0;
}
