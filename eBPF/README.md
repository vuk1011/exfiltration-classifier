# eBPF DNS monitor (CO-RE) + ML klasifikator — MONITORING ONLY

cgroup/skb egress hvata DNS upite NA IZLASKU iz svakog poda (identitet = pravi
pod, ne CoreDNS), racuna 9 feature-a i klasifikuje ih ML modelom (random forest
iz ml-classifier/). Ovo je ciste monitoring verzija: NE menja labele niti
blokira saobracaj (karantin je uklonjen).

## Build
    make
Nastaje: vmlinux.h (iz BTF-a), dns.bpf.o, dns_loader (C varijanta).

## Dashboard (opciono)
    sudo python3 -m pip install --break-system-packages rich
    sudo DASHBOARD=1 ML_DIR=../ml-classifier MODEL_PATH=../ml-classifier/model.pkl python3 dns_loader.py

## Pokretanje (preporuceno: Python + ML)
    sudo ML_DIR=../ml-classifier MODEL_PATH=../ml-classifier/model.pkl \
         python3 dns_loader.py

- Ako je model.pkl prisutan, koristi se ML skor (prag preko THRESHOLD, default 0.5).
- Ako modela nema, koristi se rezervni prag (grubo, dok ne istreniras model).

### Treniranje modela (u ml-classifier/)
    cd ../ml-classifier
    ./bin/pip install -r requirements.txt
    python app.py train <dataset.csv> --output-dir out
    # pa koristi out/model.pkl kao MODEL_PATH

## C varijanta (bez ML-a, samo rezervni prag)
    sudo ./dns_loader
C loader racuna uzi skup feature-a i koristi prag; za ML koristi Python loader.

## Sta se ispisuje
- ALARM (crveno): izvorni pod sa visokim skorom (namespace, pod, DNS ime, feature-i).
- TRANZIT (zuto): isti upit vidjen na CoreDNS-u (nije izvor; samo prikaz).
- Obicni redovi: normalan saobracaj sa skorom.

## 9 feature-a (moraju da prate trening semantiku dataseta)
len, subdomains_count, w_count, w_max, entropy,
w_max_ratio, w_count_ratio, digits_ratio, uppercase_ratio
Racunaju se nad imenom bez tacaka; "reci" su nizovi slova. Vidi extract_features.

## Napomene
- Novi podovi podignuti POSLE pokretanja loadera nece biti zakaceni; pokreni
  loader kad je lab vec gore (ili dodaj periodicni rescan).
- Ovo je monitoring: za prevenciju (karantin) vidi zasebne NetworkPolicy fajlove.
