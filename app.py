"""
app.py - Antarmuka Streamlit untuk proyek Kriptografi:
S-Box hasil Genetic Algorithm + AES-128 untuk enkripsi gambar.

Jalankan lokal :  streamlit run app.py
"""
import io
import random
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from crypto_core import (SBoxAES, entropy, is_valid_sbox, run_ga, sbox_report,
                         sensitivity_test, NPCR_IDEAL, UACI_IDEAL)

st.set_page_config(page_title="S-Box GA & Enkripsi Gambar AES", page_icon="🔐", layout="wide")

DEFAULT_SBOX_PATH = Path(__file__).parent / "sbox_default.npy"


# ---------------------------------------------------------------------
# State & cache
# ---------------------------------------------------------------------
def init_state():
    if "sbox" not in st.session_state:
        st.session_state.sbox = np.load(DEFAULT_SBOX_PATH).astype(np.uint8)
        st.session_state.sbox_source = "S-Box default (hasil GA yang sudah dihitung sebelumnya)"
        st.session_state.ga_history = None
        st.session_state.ga_info = None
    if "enc_results" not in st.session_state:
        st.session_state.enc_results = None


@st.cache_resource(show_spinner=False)
def get_aes(sbox_bytes: bytes) -> SBoxAES:
    return SBoxAES(np.frombuffer(sbox_bytes, dtype=np.uint8))


@st.cache_data(show_spinner=False)
def get_report(sbox_bytes: bytes) -> dict:
    return sbox_report(np.frombuffer(sbox_bytes, dtype=np.uint8))


def sbox_table(sbox, mode="Hex"):
    arr = np.asarray(sbox).reshape(16, 16)
    labels = [f"{i:X}" for i in range(16)]
    if mode == "Hex":
        data = [[f"{int(v):02X}" for v in row] for row in arr]
    else:
        data = [[int(v) for v in row] for row in arr]
    return pd.DataFrame(data, index=labels, columns=labels)


def parse_key_iv(key_text, iv_hex):
    key = key_text.encode("utf-8")
    if len(key) != 16:
        raise ValueError(f"Kunci harus tepat 16 byte (sekarang {len(key)} byte).")
    try:
        iv = bytes.fromhex(iv_hex.strip())
    except ValueError:
        raise ValueError("IV harus berupa heksadesimal.")
    if len(iv) != 16:
        raise ValueError(f"IV harus 32 karakter heksadesimal / 16 byte (sekarang {len(iv)} byte).")
    return key, iv


def png_bytes(im):
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def histogram_figure(original, encrypted, title):
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.5))
    names, colors = ["Red", "Green", "Blue"], ["tab:red", "tab:green", "tab:blue"]
    for ch in range(3):
        axes[0, ch].hist(original[:, :, ch].ravel(), bins=256, range=(0, 255), color=colors[ch])
        axes[0, ch].set_title(f"Original - {names[ch]}")
        axes[1, ch].hist(encrypted[:, :, ch].ravel(), bins=256, range=(0, 255), color=colors[ch])
        axes[1, ch].set_title(f"Cipher - {names[ch]}")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def sac_heatmap(sac):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(sac, vmin=0.4, vmax=0.6, cmap="coolwarm")
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{sac[i, j]:.3f}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("Output bit")
    ax.set_ylabel("Input bit")
    ax.set_title("Matriks SAC (ideal 0,5)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


init_state()

st.title("🔐 S-Box Genetic Algorithm & Enkripsi Gambar AES-128")
st.caption("Proyek Kriptografi: S-Box dibangkitkan dengan Genetic Algorithm, diuji dengan 6 kriteria, "
           "lalu dipakai pada AES-128 untuk enkripsi/dekripsi gambar.")

tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Bangkitkan S-Box (GA)", "2️⃣ Uji S-Box",
                                  "3️⃣ Enkripsi & Dekripsi Gambar", "4️⃣ Dekripsi dari file"])

# =====================================================================
# TAB 1
# =====================================================================
with tab1:
    st.subheader("S-Box aktif")
    st.info(st.session_state.sbox_source)
    if st.session_state.ga_info:
        i = st.session_state.ga_info
        st.caption(f"Seed terbaik: {i['seed']} | polinomial irreducible: {i['poly']} | fitness: {i['fitness']:.5f}")

    with st.expander("⚙️ Parameter Genetic Algorithm & bangkitkan S-Box baru", expanded=False):
        st.markdown(
            "Kromosom GA berisi parameter konstruksi **S(x) = A·inv(B·x ⊕ d) ⊕ c** di GF(2⁸): "
            "pilihan polinomial irreducible, matriks affine A dan B (8×8 invertible), serta konstanta c dan d. "
            "Fitness meminimalkan penyimpangan SAC dan BIC-SAC dari 0,5."
        )
        c1, c2, c3 = st.columns(3)
        pop = c1.slider("Ukuran populasi", 20, 120, 60, 10)
        gens = c2.slider("Generasi per restart", 20, 200, 100, 10)
        rest = c3.slider("Jumlah restart (seed berbeda)", 1, 8, 3)
        st.caption("Perkiraan waktu: beberapa detik per restart di komputer biasa; "
                   "di Streamlit Cloud bisa 2–3× lebih lambat.")
        if st.button("🚀 Jalankan GA"):
            bar = st.progress(0.0, text="Memulai GA...")

            def cb(r, rt, g, gt, best):
                if g % 5 == 0 or g == gt:
                    done = ((r - 1) * gt + g) / (rt * gt)
                    bar.progress(min(done, 1.0), text=f"Restart {r}/{rt} | generasi {g}/{gt} | fitness terbaik {best:.5f}")

            sbox, hist, info = run_ga(pop_size=pop, generations=gens, restarts=rest, progress=cb)
            bar.progress(1.0, text="Selesai")
            st.session_state.sbox = sbox
            st.session_state.sbox_source = "S-Box baru hasil GA (dibangkitkan di sesi ini)"
            st.session_state.ga_history = pd.DataFrame(hist)
            st.session_state.ga_info = info
            st.session_state.enc_results = None   # hasil enkripsi lama tidak berlaku lagi
            st.success("S-Box baru berhasil dibuat dan menjadi S-Box aktif.")
            st.rerun()

        if st.button("↩️ Kembalikan ke S-Box default"):
            for k in ("sbox", "enc_results"):
                st.session_state.pop(k, None)
            st.session_state.pop("ga_history", None)
            st.session_state.pop("ga_info", None)
            st.rerun()

    mode = st.radio("Tampilan tabel 16×16", ["Hex", "Desimal"], horizontal=True)
    st.dataframe(sbox_table(st.session_state.sbox, mode))
    cA, cB = st.columns(2)
    cA.download_button("⬇️ Unduh S-Box (CSV)", sbox_table(st.session_state.sbox, "Desimal").to_csv(),
                       file_name="sbox.csv", mime="text/csv")
    buf = io.BytesIO()
    np.save(buf, st.session_state.sbox)
    cB.download_button("⬇️ Unduh S-Box (.npy)", buf.getvalue(), file_name="sbox.npy")

    if st.session_state.ga_history is not None:
        st.subheader("Perkembangan fitness")
        st.line_chart(st.session_state.ga_history.set_index("generation")[["best_fitness", "mean_fitness"]])

# =====================================================================
# TAB 2
# =====================================================================
with tab2:
    sbox = st.session_state.sbox
    rep = get_report(sbox.tobytes())
    st.subheader("Hasil pengujian S-Box")

    def status(ok):
        return "✅" if ok else "❌"

    rows = [
        ["Balance", "Lulus" if rep["balance_ok"] else "Gagal", "Tiap bit output 128:128", status(rep["balance_ok"])],
        ["Bijective", f"{rep['unique_count']}/256 unik", "256 nilai unik", status(rep["bijective_ok"])],
        ["Nonlinearity (NL)", rep["NL"], "112", status(rep["NL"] >= 112)],
        ["SAC (rata-rata matriks)", f"{rep['SAC_mean']:.5f}", "0,5", status(abs(rep["SAC_mean"] - .5) < 0.005)],
        ["BIC-NL", rep["BIC_NL"], "112", status(rep["BIC_NL"] >= 112)],
        ["BIC-SAC (rata-rata)", f"{rep['BICSAC_mean']:.5f}", "0,5", status(abs(rep["BICSAC_mean"] - .5) < 0.005)],
        ["LAP", f"{rep['LAP']:.5f}", "→ 0 (0,0625 = AES)", status(rep["LAP"] <= 0.0625)],
        ["DAP", f"{rep['DAP']:.6f}", "→ 0 (0,015625 = AES)", status(rep["DAP"] <= 4 / 256)],
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Pengujian", "Hasil", "Target / ideal", "Status"]), hide_index=True)
    st.caption(f"Deviasi ketat (rata-rata |nilai − 0,5|): SAC = {rep['SAC_dev']:.5f}, BIC-SAC = {rep['BICSAC_dev']:.5f}. "
               "Definisi LAP mengikuti paper: max|#{x: x·a = S(x)·b}/2ⁿ − 1/2|. "
               "DAP terbaik yang dikenal untuk S-Box 8-bit bijektif adalah 4/256 (S-Box AES).")

    comp = pd.DataFrame({
        "Kriteria": ["NL", "SAC", "BIC-NL", "BIC-SAC", "LAP", "DAP"],
        "S-Box aktif": [rep["NL"], round(rep["SAC_mean"], 5), rep["BIC_NL"], round(rep["BICSAC_mean"], 5),
                        round(rep["LAP"], 5), round(rep["DAP"], 6)],
        "AES": [112, 0.50488, 112, 0.50460, 0.0625, 0.015625],
        "S-box44 (paper Springer)": [112, 0.50073, 112, 0.50237, 0.0625, 0.015625],
    })
    st.markdown("**Perbandingan dengan S-Box AES dan paper**")
    st.dataframe(comp, hide_index=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(sac_heatmap(rep["SAC_matrix"]))
    with c2:
        st.markdown("**Uji balance per bit output**")
        st.dataframe(pd.DataFrame(rep["balance_table"]), hide_index=True)

# =====================================================================
# TAB 3
# =====================================================================
with tab3:
    st.subheader("Enkripsi & dekripsi gambar dengan AES-128 + S-Box aktif")
    st.caption("Mode: CBC dua arah (maju + mundur) sehingga perubahan 1 byte di mana pun menyebar ke seluruh ciphertext.")
    c1, c2, c3 = st.columns([2, 3, 1])
    key_text = c1.text_input("Kunci (16 karakter)", value="GA-SBox-Demo-123", max_chars=16)
    iv_hex = c2.text_input("IV (32 karakter hex)", value="00112233445566778899aabbccddeeff")
    n_trials = c3.slider("Percobaan NPCR/UACI", 4, 20, 10)

    st.markdown("Dataset paper: `cameraman`, `lena_color_512`, `livingroom`, `mandril_color`, "
                "`pirate`, `woman_blonde`, `woman_darkhair`.")
    uploads = st.file_uploader("Unggah satu atau beberapa gambar", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                               accept_multiple_files=True)
    st.caption("Estimasi waktu: gambar 512×512 RGB ≈ 2–4 detik per enkripsi; "
               "total ≈ (percobaan + 1) × waktu enkripsi per gambar.")

    if st.button("🔒 Enkripsi, dekripsi & uji", disabled=not uploads):
        try:
            key, iv = parse_key_iv(key_text, iv_hex)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        aes = get_aes(st.session_state.sbox.tobytes())
        results = []
        bar = st.progress(0.0, text="Memproses...")
        for idx, f in enumerate(uploads):
            bar.progress(idx / len(uploads), text=f"Memproses {f.name} ({idx + 1}/{len(uploads)})")
            img = Image.open(f).convert("RGB")
            raw = img.tobytes()
            cipher = aes.encrypt(raw, key, iv)
            recovered = aes.decrypt(cipher, key, iv)
            ok = recovered == raw
            cipher_img = Image.frombytes("RGB", img.size, cipher[:len(raw)])
            dec_img = Image.frombytes("RGB", img.size, recovered)
            sens = sensitivity_test(aes, raw, cipher, key, iv, n_trials=n_trials, rng=random.Random(idx))
            results.append({"name": f.name, "img": img, "cipher_img": cipher_img, "dec_img": dec_img,
                            "cipher": cipher, "ok": ok, "w": img.width, "h": img.height,
                            "H_plain": entropy(raw), "H_cipher": entropy(cipher[:len(raw)]), "sens": sens})
        bar.progress(1.0, text="Selesai")
        st.session_state.enc_results = results

    results = st.session_state.enc_results
    if results:
        summary = pd.DataFrame([{
            "Gambar": r["name"], "Ukuran": f"{r['w']}×{r['h']}",
            "Entropy asli": round(r["H_plain"], 4), "Entropy cipher (ideal 8)": round(r["H_cipher"], 5),
            f"NPCR % (ideal {NPCR_IDEAL:.2f})": round(r["sens"]["NPCR_mean"], 4),
            f"UACI % (ideal {UACI_IDEAL:.2f})": round(r["sens"]["UACI_mean"], 4),
            "NPCR lulus uji Wu": r["sens"]["NPCR_lulus"], "UACI lulus uji Wu": r["sens"]["UACI_lulus"],
            "Dekripsi sama persis": "✅" if r["ok"] else "❌",
        } for r in results])
        st.subheader("Ringkasan")
        st.dataframe(summary, hide_index=True)
        st.download_button("⬇️ Unduh ringkasan (CSV)", summary.to_csv(index=False),
                           file_name="hasil_enkripsi.csv", mime="text/csv")

        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            for r in results:
                stem = Path(r["name"]).stem
                for label, im in (("encrypted", r["cipher_img"]), ("decrypted", r["dec_img"])):
                    z.writestr(f"{stem}_{label}.png", png_bytes(im))
                z.writestr(f"{stem}.bin", r["cipher"])
        st.download_button("⬇️ Unduh semua hasil (ZIP)", zbuf.getvalue(), file_name="hasil_enkripsi.zip")

        st.subheader("Detail per gambar")
        for i, r in enumerate(results):
            with st.expander(f"{r['name']}  —  NPCR {r['sens']['NPCR_mean']:.4f}% | UACI {r['sens']['UACI_mean']:.4f}%"):
                a, b, c = st.columns(3)
                a.image(r["img"], caption="Gambar asli")
                b.image(r["cipher_img"], caption="Gambar terenkripsi")
                c.image(r["dec_img"], caption="Hasil dekripsi " + ("(identik)" if r["ok"] else "(BERBEDA!)"))
                fig = histogram_figure(np.asarray(r["img"]), np.asarray(r["cipher_img"]), f"Histogram: {r['name']}")
                st.pyplot(fig)
                plt.close(fig)
                s = r["sens"]
                st.markdown(
                    f"**NPCR**: rata-rata {s['NPCR_mean']:.4f}%, minimum {s['NPCR_min']:.4f}% "
                    f"(batas bawah Wu et al. {s['NPCR_batas_bawah']:.4f}%) · lulus {s['NPCR_lulus']}  \n"
                    f"**UACI**: rata-rata {s['UACI_mean']:.4f}%, rentang {s['UACI_min']:.4f}–{s['UACI_max']:.4f}% "
                    f"(batas Wu et al. {s['UACI_batas_bawah']:.4f}–{s['UACI_batas_atas']:.4f}%) · lulus {s['UACI_lulus']}  \n"
                    f"Percobaan: flip 1 bit di byte pertama, terakhir, dan {s['trials'] - 2} posisi acak."
                )
                st.download_button("⬇️ Gambar terenkripsi (PNG)", png_bytes(r["cipher_img"]),
                                   file_name=f"{Path(r['name']).stem}_encrypted.png", mime="image/png", key=f"dl_{i}")

# =====================================================================
# TAB 4
# =====================================================================
with tab4:
    st.subheader("Dekripsi dari file ciphertext (.bin)")
    st.caption("Gunakan file .bin dari ZIP hasil enkripsi, S-Box aktif yang sama, serta kunci dan IV yang sama.")
    d1, d2 = st.columns(2)
    bin_file = d1.file_uploader("File ciphertext (.bin)", type=["bin"])
    dk = d2.text_input("Kunci (16 karakter)", value="GA-SBox-Demo-123", max_chars=16, key="dec_key")
    di = d2.text_input("IV (32 karakter hex)", value="00112233445566778899aabbccddeeff", key="dec_iv")
    w = d1.number_input("Lebar gambar (px)", min_value=1, value=512, step=1)
    h = d1.number_input("Tinggi gambar (px)", min_value=1, value=512, step=1)
    if st.button("🔓 Dekripsi", disabled=bin_file is None):
        try:
            key, iv = parse_key_iv(dk, di)
            aes = get_aes(st.session_state.sbox.tobytes())
            raw = aes.decrypt(bin_file.getvalue(), key, iv)
            if len(raw) != int(w) * int(h) * 3:
                st.error(f"Ukuran data hasil dekripsi ({len(raw)} byte) tidak sama dengan {int(w)}×{int(h)}×3. "
                         "Periksa lebar dan tinggi.")
            else:
                st.image(Image.frombytes("RGB", (int(w), int(h)), raw), caption="Hasil dekripsi")
        except ValueError as e:
            st.error(f"Dekripsi gagal: {e}")
