"""
crypto_core.py
Logika inti proyek (tanpa Streamlit) supaya mudah diuji:
  1. Pembuatan S-Box dengan Genetic Algorithm (parameter konstruksi affine di atas invers GF(2^8))
  2. Pengujian S-Box: balance, bijective, NL, SAC, BIC-NL, BIC-SAC, LAP, DAP
  3. AES-128 dengan S-Box kustom, mode CBC dua arah (difusi ke seluruh ciphertext)
  4. Entropy, NPCR, UACI + batas kritis Wu et al.
"""
import random
import numpy as np
from scipy.stats import norm

# =====================================================================
# 1. GF(2^8) dan GA
# =====================================================================
_PAR = [bin(i).count("1") & 1 for i in range(256)]
PARITY = np.array(_PAR, dtype=np.int8)
X = np.arange(256)


def gf_mul_poly(a, b, poly):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        if a & 0x100:
            a ^= poly
        b >>= 1
    return r


def _pmod(a, b):
    while a.bit_length() >= b.bit_length():
        a ^= b << (a.bit_length() - b.bit_length())
    return a


def _irreducible(poly):
    return all(_pmod(poly, d) != 0 for d in range(2, 32))


IRRED_POLYS = [p for p in range(0x100, 0x200) if _irreducible(p)]
assert len(IRRED_POLYS) == 30

_INV_TABLES = None


def _inv_tables():
    global _INV_TABLES
    if _INV_TABLES is None:
        tabs = []
        for poly in IRRED_POLYS:
            t = np.zeros(256, dtype=np.int64)
            for a in range(1, 256):
                r = 1
                for _ in range(254):          # a^254 = a^-1
                    r = gf_mul_poly(r, a, poly)
                t[a] = r
            tabs.append(t)
        _INV_TABLES = tabs
    return _INV_TABLES


def mat_rank(rows):
    rows = list(rows)
    r = 0
    for bit in range(8):
        piv = next((i for i in range(r, 8) if rows[i] >> bit & 1), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(8):
            if i != r and rows[i] >> bit & 1:
                rows[i] ^= rows[r]
        r += 1
    return r


def _rand_inv_matrix(rng):
    while True:
        m = [rng.randrange(1, 256) for _ in range(8)]
        if mat_rank(m) == 8:
            return m


def _linear_table(m):
    return np.array([sum(_PAR[m[i] & v] << i for i in range(8)) for v in range(256)], dtype=np.int64)


def build_sbox(g):
    """S(x) = A * inv(B*x XOR d) XOR c ;  g = [indeks_polinomial, A, B, c, d]"""
    poly, A, B, c, d = g
    inv = _inv_tables()[poly]
    return (_linear_table(A)[inv[_linear_table(B)[X] ^ d]] ^ c).astype(np.uint8)


def _sac_bic_fast(s):
    s = s.astype(np.int64)
    sac = np.zeros((8, 8))
    bic = []
    for i in range(8):
        ch = s ^ s[X ^ (1 << i)]
        bits = [(ch >> j) & 1 for j in range(8)]
        for j in range(8):
            sac[i, j] = bits[j].mean()
            for k in range(j + 1, 8):
                bic.append((bits[j] ^ bits[k]).mean())
    return sac, np.array(bic)


def fitness(g):
    sac, bic = _sac_bic_fast(build_sbox(g))
    strict = (np.abs(sac - .5).mean() + np.abs(bic - .5).mean()) / 2
    paper = abs(sac.mean() - .5) + abs(bic.mean() - .5)
    return float(strict + 0.5 * paper + 0.1 * np.abs(sac - .5).max())


def _copy(g):
    return [g[0], list(g[1]), list(g[2]), g[3], g[4]]


def _flip_keep_inv(m, rng):
    for _ in range(20):
        n = list(m)
        n[rng.randrange(8)] ^= 1 << rng.randrange(8)
        if mat_rank(n) == 8:
            return n
    return m


def _mutate(g, rng):
    g = _copy(g)
    for _ in range(rng.randint(1, 3)):
        t = rng.random()
        if t < .10:
            g[0] = rng.randrange(30)
        elif t < .45:
            g[1] = _flip_keep_inv(g[1], rng)
        elif t < .80:
            g[2] = _flip_keep_inv(g[2], rng)
        elif t < .90:
            g[3] ^= 1 << rng.randrange(8)
        else:
            g[4] ^= 1 << rng.randrange(8)
    return g


def _crossover(a, b, rng):
    child = [rng.choice([a[0], b[0]]),
             [rng.choice(p) for p in zip(a[1], b[1])],
             [rng.choice(p) for p in zip(a[2], b[2])],
             rng.choice([a[3], b[3]]), rng.choice([a[4], b[4]])]
    if mat_rank(child[1]) < 8 or mat_rank(child[2]) < 8:
        return _copy(a)
    return child


def run_ga(pop_size=80, generations=150, restarts=8, base_seed=1,
           elite=4, crossover_rate=0.8, mutation_rate=0.6, tournament=3,
           progress=None):
    """
    Jalankan GA beberapa kali (restart). progress(restart_ke, total_restart, generasi, total_gen, best_fitness)
    Return: (sbox uint8[256], history list[dict], info dict)
    """
    _inv_tables()  # siapkan tabel invers
    overall_best, overall_score, overall_seed, all_hist = None, float("inf"), None, []
    for r in range(restarts):
        seed = base_seed + r
        rng = random.Random(seed)
        pop = [[rng.randrange(30), _rand_inv_matrix(rng), _rand_inv_matrix(rng),
                rng.randrange(256), rng.randrange(256)] for _ in range(pop_size)]
        best, best_score, hist = None, float("inf"), []
        for gen in range(generations):
            scores = [fitness(g) for g in pop]
            order = np.argsort(scores)
            pop = [pop[i] for i in order]
            scores = [scores[i] for i in order]
            if scores[0] < best_score:
                best_score, best = scores[0], _copy(pop[0])
            hist.append({"restart": seed, "generation": gen + 1,
                         "best_fitness": float(scores[0]), "mean_fitness": float(np.mean(scores))})
            if progress:
                progress(r + 1, restarts, gen + 1, generations, best_score)
            new = [_copy(pop[i]) for i in range(elite)]
            while len(new) < pop_size:
                def pick():
                    ids = rng.sample(range(pop_size), tournament)
                    return pop[min(ids, key=lambda i: scores[i])]
                p1, p2 = pick(), pick()
                child = _crossover(p1, p2, rng) if rng.random() < crossover_rate else _copy(p1)
                if rng.random() < mutation_rate:
                    child = _mutate(child, rng)
                new.append(child)
            pop = new
        all_hist += hist
        if best_score < overall_score:
            overall_score, overall_best, overall_seed = best_score, best, seed
    sbox = build_sbox(overall_best)
    assert is_valid_sbox(sbox)
    history = [h for h in all_hist if h["restart"] == overall_seed]
    info = {"seed": overall_seed, "poly": bin(IRRED_POLYS[overall_best[0]]), "fitness": overall_score}
    return sbox, history, info


# =====================================================================
# 2. Pengujian S-Box
# =====================================================================
def is_valid_sbox(s):
    s = np.asarray(s)
    return len(s) == 256 and np.array_equal(np.sort(s), np.arange(256))


_H = np.array([[1 - 2 * _PAR[a & b] for b in range(256)] for a in range(256)], dtype=np.int32)
_COMP = np.array([[1 - 2 * _PAR[v & m] for m in range(1, 256)] for v in range(256)], dtype=np.int32)


def sbox_report(sbox):
    """Semua pengujian S-Box. Return dict nilai + detail."""
    s = np.asarray(sbox, dtype=np.int64)
    out = {}
    # Balance & bijective
    bal_rows = []
    for bit in range(8):
        ones = int(((s >> bit) & 1).sum())
        bal_rows.append({"Output bit": bit, "Jumlah 0": 256 - ones, "Jumlah 1": ones,
                         "Seimbang 128:128": ones == 128})
    out["balance_table"] = bal_rows
    out["balance_ok"] = all(r["Seimbang 128:128"] for r in bal_rows)
    out["unique_count"] = int(len(np.unique(s)))
    out["bijective_ok"] = out["unique_count"] == 256 and is_valid_sbox(s)
    # Walsh untuk 255 komponen
    W = np.abs(_H @ _COMP[s]).max(axis=0)        # max |Walsh| per komponen
    nl_comp = 128 - W // 2
    out["NL"] = int(nl_comp.min())
    pair_masks = [(1 << i) | (1 << j) for i in range(8) for j in range(i + 1, 8)]
    out["BIC_NL"] = int(min(nl_comp[m - 1] for m in pair_masks))
    out["LAP"] = float(W.max()) / 2 / 256        # definisi paper: max|#/2^n - 1/2|
    # SAC & BIC-SAC
    sac, bic = _sac_bic_fast(s.astype(np.uint8))
    out["SAC_matrix"] = sac
    out["SAC_mean"] = float(sac.mean())
    out["SAC_dev"] = float(np.abs(sac - .5).mean())
    out["BICSAC_mean"] = float(bic.mean())
    out["BICSAC_dev"] = float(np.abs(bic - .5).mean())
    # DAP
    mx = 0
    for dx in range(1, 256):
        dy = s ^ s[X ^ dx]
        mx = max(mx, int(np.bincount(dy, minlength=256).max()))
    out["DU"] = mx
    out["DAP"] = mx / 256.0
    return out


# =====================================================================
# 3. AES-128 dengan S-Box kustom + mode CBC dua arah
# =====================================================================
def _gf_mul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = ((a << 1) ^ (0x11B if a & 0x80 else 0)) & 0xFF
        b >>= 1
    return r


_ISR_IDX = np.array([0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3])
_MUL = {m: np.array([_gf_mul(i, m) for i in range(256)], dtype=np.uint8) for m in (9, 11, 13, 14)}


class SBoxAES:
    """AES-128 yang SubBytes & key-expansion-nya memakai S-Box yang diberikan."""

    def __init__(self, sbox):
        if not is_valid_sbox(sbox):
            raise ValueError("S-Box harus permutasi 0..255")
        self.S = [int(v) for v in sbox]
        self.sbox = np.array(self.S, dtype=np.uint8)
        self.inv = np.empty(256, dtype=np.uint8)
        self.inv[self.sbox] = np.arange(256, dtype=np.uint8)
        m2 = [_gf_mul(i, 2) for i in range(256)]
        m3 = [_gf_mul(i, 3) for i in range(256)]
        T0, T1, T2, T3 = [], [], [], []
        for x in range(256):
            v = self.S[x]
            v2, v3 = m2[v], m3[v]
            T0.append((v2 << 24) | (v << 16) | (v << 8) | v3)
            T1.append((v3 << 24) | (v2 << 16) | (v << 8) | v)
            T2.append((v << 24) | (v3 << 16) | (v2 << 8) | v)
            T3.append((v << 24) | (v << 16) | (v3 << 8) | v2)
        self.T = (T0, T1, T2, T3)

    # ---- key schedule ----
    def expand_key(self, key):
        if len(key) != 16:
            raise ValueError("Kunci harus 16 byte")
        S = self.S
        words = [list(key[i:i + 4]) for i in range(0, 16, 4)]
        rcon = [0, 1, 2, 4, 8, 16, 32, 64, 128, 0x1B, 0x36]
        for i in range(4, 44):
            t = words[i - 1].copy()
            if i % 4 == 0:
                t = t[1:] + t[:1]
                t = [S[b] for b in t]
                t[0] ^= rcon[i // 4]
            words.append([words[i - 4][j] ^ t[j] for j in range(4)])
        return [sum(words[4 * r:4 * r + 4], []) for r in range(11)]

    def _rk_words(self, key):
        return [[(rk[4 * c] << 24) | (rk[4 * c + 1] << 16) | (rk[4 * c + 2] << 8) | rk[4 * c + 3]
                 for c in range(4)] for rk in self.expand_key(key)]

    # ---- enkripsi satu blok (T-table), state = 4 word 32-bit ----
    def _enc_words(self, w0, w1, w2, w3, rk):
        S = self.S
        T0, T1, T2, T3 = self.T
        k = rk[0]
        w0 ^= k[0]; w1 ^= k[1]; w2 ^= k[2]; w3 ^= k[3]
        for r in range(1, 10):
            k = rk[r]
            n0 = T0[w0 >> 24] ^ T1[(w1 >> 16) & 255] ^ T2[(w2 >> 8) & 255] ^ T3[w3 & 255] ^ k[0]
            n1 = T0[w1 >> 24] ^ T1[(w2 >> 16) & 255] ^ T2[(w3 >> 8) & 255] ^ T3[w0 & 255] ^ k[1]
            n2 = T0[w2 >> 24] ^ T1[(w3 >> 16) & 255] ^ T2[(w0 >> 8) & 255] ^ T3[w1 & 255] ^ k[2]
            n3 = T0[w3 >> 24] ^ T1[(w0 >> 16) & 255] ^ T2[(w1 >> 8) & 255] ^ T3[w2 & 255] ^ k[3]
            w0, w1, w2, w3 = n0, n1, n2, n3
        k = rk[10]
        o0 = ((S[w0 >> 24] << 24) | (S[(w1 >> 16) & 255] << 16) | (S[(w2 >> 8) & 255] << 8) | S[w3 & 255]) ^ k[0]
        o1 = ((S[w1 >> 24] << 24) | (S[(w2 >> 16) & 255] << 16) | (S[(w3 >> 8) & 255] << 8) | S[w0 & 255]) ^ k[1]
        o2 = ((S[w2 >> 24] << 24) | (S[(w3 >> 16) & 255] << 16) | (S[(w0 >> 8) & 255] << 8) | S[w1 & 255]) ^ k[2]
        o3 = ((S[w3 >> 24] << 24) | (S[(w0 >> 16) & 255] << 16) | (S[(w1 >> 8) & 255] << 8) | S[w2 & 255]) ^ k[3]
        return o0, o1, o2, o3

    def encrypt_block(self, block16, key):
        w = tuple(int(x) for x in np.frombuffer(block16, dtype=">u4"))
        return np.array(self._enc_words(*w, self._rk_words(key)), dtype=">u4").tobytes()

    # ---- dekripsi semua blok sekaligus (NumPy) ----
    def _dec_blocks(self, blocks, rks):
        rk = [np.array(r, dtype=np.uint8) for r in rks]
        s = blocks ^ rk[10]
        s = self.inv[s[:, _ISR_IDX]]
        for r in range(9, 0, -1):
            s = s ^ rk[r]
            out = np.empty_like(s)
            for c in range(4):
                a0, a1, a2, a3 = (s[:, 4 * c + i] for i in range(4))
                out[:, 4 * c + 0] = _MUL[14][a0] ^ _MUL[11][a1] ^ _MUL[13][a2] ^ _MUL[9][a3]
                out[:, 4 * c + 1] = _MUL[9][a0] ^ _MUL[14][a1] ^ _MUL[11][a2] ^ _MUL[13][a3]
                out[:, 4 * c + 2] = _MUL[13][a0] ^ _MUL[9][a1] ^ _MUL[14][a2] ^ _MUL[11][a3]
                out[:, 4 * c + 3] = _MUL[11][a0] ^ _MUL[13][a1] ^ _MUL[9][a2] ^ _MUL[14][a3]
            s = self.inv[out[:, _ISR_IDX]]
        return s ^ rk[0]

    # ---- mode CBC dua arah ----
    def _chain(self, blocks, rk, iv_words):
        p0, p1, p2, p3 = iv_words
        out = []
        enc = self._enc_words
        for b0, b1, b2, b3 in blocks:
            p0, p1, p2, p3 = enc(b0 ^ p0, b1 ^ p1, b2 ^ p2, b3 ^ p3, rk)
            out.append((p0, p1, p2, p3))
        return out

    def encrypt(self, data, key, iv):
        if len(iv) != 16:
            raise ValueError("IV harus 16 byte")
        rk = self._rk_words(key)
        pad = 16 - (len(data) % 16)
        padded = data + bytes([pad]) * pad
        w = np.frombuffer(padded, dtype=">u4").astype(np.int64).reshape(-1, 4)
        blocks = [tuple(int(x) for x in row) for row in w]
        iv1 = tuple(int(x) for x in np.frombuffer(iv, dtype=">u4"))
        iv2 = self._enc_words(*iv1, rk)
        c = self._chain(blocks, rk, iv1)                    # pass maju
        d = self._chain(c[::-1], rk, iv2)[::-1]             # pass mundur
        return np.array(d, dtype=">u4").tobytes()

    def decrypt(self, ct, key, iv):
        if len(iv) != 16 or len(ct) % 16 or len(ct) == 0:
            raise ValueError("IV harus 16 byte dan ciphertext kelipatan 16 byte")
        rks = self.expand_key(key)
        rk = self._rk_words(key)
        iv1 = tuple(int(x) for x in np.frombuffer(iv, dtype=">u4"))
        iv2 = np.frombuffer(np.array(self._enc_words(*iv1, rk), dtype=">u4").tobytes(), dtype=np.uint8)
        D = np.frombuffer(ct, dtype=np.uint8).reshape(-1, 16)
        C = self._dec_blocks(D, rks) ^ np.vstack([D[1:], iv2[None, :]])
        P = self._dec_blocks(C, rks) ^ np.vstack([np.frombuffer(iv, dtype=np.uint8)[None, :], C[:-1]])
        padded = P.tobytes()
        pad = padded[-1]
        if pad < 1 or pad > 16 or padded[-pad:] != bytes([pad]) * pad:
            raise ValueError("Padding tidak valid (kunci/IV salah atau ciphertext rusak)")
        return padded[:-pad]


# =====================================================================
# 4. Metrik citra
# =====================================================================
NPCR_IDEAL = 255 / 256 * 100
UACI_IDEAL = (255 + 2) / (3 * 255 + 3) * 100


def entropy(data):
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / len(data)
    return float(-np.sum(p * np.log2(p)))


def npcr_uaci(a_bytes, b_bytes):
    a = np.frombuffer(a_bytes, dtype=np.uint8).astype(np.int16)
    b = np.frombuffer(b_bytes, dtype=np.uint8).astype(np.int16)
    if len(a) != len(b):
        raise ValueError("Panjang data harus sama")
    return float(np.mean(a != b) * 100), float(np.mean(np.abs(a - b) / 255.0) * 100)


def critical_values(L, alpha=0.05, F=255):
    """Batas kritis Wu et al. (2011): NPCR >= batas bawah; UACI di dalam [lo, hi]."""
    npcr_min = (F - norm.ppf(1 - alpha) * np.sqrt(F / L)) / (F + 1) * 100
    mu = (F + 2) / (3 * F + 3)
    sd = np.sqrt((F + 2) * (F ** 2 + 2 * F + 3) / (18 * (F + 1) ** 2 * F * L))
    z = norm.ppf(1 - alpha / 2)
    return float(npcr_min), float((mu - z * sd) * 100), float((mu + z * sd) * 100)


def sensitivity_test(aes, raw, cipher, key, iv, n_trials=10, rng=None):
    """Flip 1 bit di byte pertama, terakhir, dan (n_trials-2) posisi acak; bandingkan ciphertext (tanpa padding)."""
    rng = rng or random.Random(0)
    n = len(raw)
    positions = [0, n - 1] + [rng.randrange(n) for _ in range(max(0, n_trials - 2))]
    npcrs, uacis = [], []
    for pos in positions:
        mod = bytearray(raw)
        mod[pos] ^= 1
        c2 = aes.encrypt(bytes(mod), key, iv)
        a, b = npcr_uaci(cipher[:n], c2[:n])
        npcrs.append(a)
        uacis.append(b)
    nmin, ulo, uhi = critical_values(n)
    return {"trials": len(positions),
            "NPCR_mean": float(np.mean(npcrs)), "NPCR_min": min(npcrs),
            "UACI_mean": float(np.mean(uacis)), "UACI_min": min(uacis), "UACI_max": max(uacis),
            "NPCR_batas_bawah": nmin, "UACI_batas_bawah": ulo, "UACI_batas_atas": uhi,
            "NPCR_lulus": f"{sum(x >= nmin for x in npcrs)}/{len(npcrs)}",
            "UACI_lulus": f"{sum(ulo <= x <= uhi for x in uacis)}/{len(uacis)}"}
