# FIR Filter Designer (Web)

A browser-based FIR filter playground powered by **PyScript** (NumPy/SciPy/Matplotlib in the browser). Design LP/HP/BP/BS filters with common windows, visualize responses, and export coefficients—no installs needed.

**Live site:** https://epane.github.io/fir-filter-designer-web/  
*(Replace with your Pages URL if different.)*

---

## Features
- FIR types: **Lowpass**, **Highpass**, **Bandpass**, **Bandstop**
- Window functions: Hamming, Hann, Blackman, Blackman-Harris, Nuttall, FlatTop, etc.
- Plots: impulse/step, window (time), filter magnitude (±phase), window magnitude, example signal (raw vs filtered)
- **Export** coefficients (CSV) + metadata (TXT) and **Export** plots (PNG)
- Runs entirely **client-side** (no server); works on GitHub Pages
- Built-in **Help** modal with quick usage steps

---

## Quick Start

### Online
1. Open the live site.
2. Wait for “Loading Python runtime…” to finish (first load pulls NumPy/SciPy/Matplotlib).
3. Set parameters → click **Design Filter**.

### Local (optional)
You can run it locally with any static server:
```bash
python -m http.server 8000
# then open http://localhost:8000
