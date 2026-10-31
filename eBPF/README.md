# eBPF DNS monitor (CO-RE) - faza 3: cgroup/skb egress + identitet poda

## Sta je novo u odnosu na udp_sendmsg verziju
- Program je cgroup_skb/egress: kaci se na cgroup SVAKOG poda, pa se okida
  NA IZLASKU paketa iz poda, PRE CoreDNS-a. Zato je identitet uvek pravi pod,
  a ne coredns.
- Loader prolazi /sys/fs/cgroup/kubepods/<qos>/pod<UID>, kaci program na svaki,
  i mapira cgroup_id (inode cgroup dir-a) -> namespace/pod preko /var/log/pods.
- Dodat TXT-upit flag (APT34 / Helix Kitten voli TXT kanal).
- Dodata "quarantine" mapa: upises cgroup_id poda i kernel odbacuje njegov DNS.
  Karantin je zakomentarisan u handle() dok testiras monitoring.

## Build i pokretanje
    make
    sudo ./dns_loader

Pri startu ispise koje je podove zakacio. Pa pusti corrupt pod; alarm sada
pokazuje PRAVI pod (npr. investment-fund / employee-corrupt-service-...).

## Karantin (kad budes spremna)
U handle(), otkomentarisi quarantine_pod(e->cgroup_id);
Tada ce svaki sledeci DNS tog poda biti odbacen u kernelu (return 0).

## Napomene
- Novi podovi koji se podignu POSLE pokretanja loadera nece biti zakaceni;
  za demo pokreni loader kad je corrupt pod vec gore. (Kasnije: periodicni
  rescan cgroup-ova.)
- cgroup_skb egress na cgroup v2 daje paket od IP headera (bez eth).
