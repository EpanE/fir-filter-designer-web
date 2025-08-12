# app.py — FIR Filter Designer (PyScript / Pyodide)

from js import document, URL, Blob, console
import asyncio, base64
import numpy as np
from scipy.signal import firwin, freqz, get_window, lfilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from pyodide.ffi import create_proxy  # <-- keep event handlers alive

# Optional Pillow (falls back to individual PNGs if missing)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
    console.log("Pillow not available; Export Plots will save individual PNGs.")

window_map = {
    "Hamming": "hamming", "Rectangular": "boxcar", "Triangular": "triang",
    "Bartlett": "bartlett", "Sine": "sine", "Hann": "hann", "Blackman": "blackman",
    "Nuttall": "nuttall", "BlackmanNuttall": "blackmannuttall",
    "BlackmanHarris": "blackmanharris", "FlatTop": "flattop",
}
window_tooltips = {
    "Hamming": "Hamming: Good frequency resolution, reduced side lobes",
    "Rectangular": "Rectangular (Boxcar): Sharpest time domain, worst frequency domain",
    "Triangular": "Triangular: Better sidelobe rejection than rectangular",
    "Bartlett": "Bartlett: Triangular window that touches zero at both ends",
    "Sine": "Sine: Smooth window using sine shape",
    "Hann": "Hann: Basic cosine window, balanced performance",
    "Blackman": "Blackman: Higher sidelobe attenuation",
    "Nuttall": "Nuttall: Low sidelobe leakage",
    "BlackmanNuttall": "Blackman-Nuttall: Improved Blackman variation",
    "BlackmanHarris": "Blackman-Harris: Similar to Blackman-Nuttall",
    "FlatTop": "FlatTop: Very low passband ripple",
}

current = None
_PROXIES = {}     # <-- strong refs to event proxies
_EVENTS_BOUND = False

def set_status(msg: str) -> None:
    try:
        document.getElementById("status").innerText = msg
    except Exception as e:
        console.log(f"Status update failed: {e}")

def _img_url_from_fig(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def render_to_imgs(figs) -> None:
    ids = ["plot1", "plot2", "plot3", "plot4", "plot5"]
    for i, fig in enumerate(figs[: len(ids)]):
        document.getElementById(ids[i]).src = _img_url_from_fig(fig)
    document.getElementById("plots").style.display = "grid"

def download_bytes(filename: str, data: bytes, mime: str) -> None:
    blob = Blob.new([data], {"type": mime})
    href = URL.createObjectURL(blob)
    a = document.createElement("a"); a.href = href; a.download = filename; a.click()
    URL.revokeObjectURL(href)

def update_tooltip(*_) -> None:
    try:
        label = document.getElementById("wtype").value
        document.getElementById("wtip").innerText = window_tooltips.get(label, "")
    except Exception as e:
        console.log(f"Tooltip error: {e}")

def reset_defaults(event=None) -> None:
    document.getElementById("fs").value = "100"
    document.getElementById("numtaps").value = "64"
    document.getElementById("shift").value = "0"
    document.getElementById("lowcut").value = "10"
    document.getElementById("highcut").value = "20"
    document.getElementById("ftype").value = "Lowpass"
    document.getElementById("wtype").value = "Hamming"
    document.getElementById("response").value = "Impulse"
    document.getElementById("showPhase").checked = False
    update_tooltip()
    set_status("Reset to defaults. Ready.")

def design(event=None) -> None:
    global current
    try:
        # Close previous figs to avoid buildup
        if current and "figs" in current:
            for _f in current["figs"]:
                try: plt.close(_f)
                except Exception: pass

        set_status("Designing filter…")

        fs = float(document.getElementById("fs").value)
        numtaps = int(document.getElementById("numtaps").value)
        shift = int(document.getElementById("shift").value)
        lowcut = float(document.getElementById("lowcut").value)
        highcut = float(document.getElementById("highcut").value)
        ftype = document.getElementById("ftype").value
        wlabel = document.getElementById("wtype").value
        wtype = window_map.get(wlabel, "hamming")
        response = document.getElementById("response").value
        show_phase = bool(document.getElementById("showPhase").checked)

        if fs <= 0: raise ValueError("Sampling frequency must be positive.")
        if numtaps < 2: raise ValueError("Filter length should be 2 or greater.")
        if lowcut <= 0 or highcut <= 0: raise ValueError("Frequencies must be > 0 Hz.")
        nyq = fs / 2
        if highcut >= nyq or lowcut >= nyq: raise ValueError(f"Cutoffs must be < Nyquist ({nyq} Hz).")
        if ftype in ("Bandpass","Bandstop") and lowcut >= highcut:
            raise ValueError("Low frequency must be less than high frequency for band filters.")

        if ftype == "Lowpass":
            taps = firwin(numtaps, cutoff=highcut, window=wtype, fs=fs, pass_zero="lowpass")
        elif ftype == "Highpass":
            taps = firwin(numtaps, cutoff=lowcut, window=wtype, fs=fs, pass_zero="highpass")
        elif ftype == "Bandpass":
            taps = firwin(numtaps, cutoff=[lowcut, highcut], window=wtype, fs=fs, pass_zero=False)
        else:  # Bandstop
            taps = firwin(numtaps, cutoff=[lowcut, highcut], window=wtype, fs=fs, pass_zero=True)

        if shift != 0:
            n = np.arange(numtaps)
            taps = (taps * np.exp(1j * 2 * np.pi * shift * n / numtaps)).real

        window_vals = get_window(wtype, numtaps, fftbins=False)

        t = np.arange(numtaps) / fs
        step_input = np.ones(numtaps * 3)
        step_response = lfilter(taps, [1.0], step_input)
        step_time = np.arange(len(step_response)) / fs

        w, h_filter = freqz(taps, worN=8192, fs=fs)
        w_win, h_window = freqz(window_vals, worN=8192, fs=fs)

        dur = 2.0
        N = int(fs * dur); tt = np.arange(N) / fs
        rng = np.random.default_rng(0)
        x = 0.9*np.sin(2*np.pi*5*tt) + 0.4*np.sin(2*np.pi*25*tt) + 0.05*rng.normal(size=N)
        y = lfilter(taps, [1.0], x)

        plt.style.use("default")
        figs = []

        f1, a1 = plt.subplots(figsize=(5, 3)); f1.patch.set_facecolor("white")
        if response in ("Impulse","Both"):
            a1.stem(t, taps, basefmt=" ")
            a1.set_title("Impulse Response")
        else:
            a1.plot(step_time[: numtaps*2], step_response[: numtaps*2])
            a1.set_title("Step Response")
        a1.set_xlabel("Time (s)"); a1.set_ylabel("Amplitude"); a1.grid(True, alpha=0.3)
        figs.append(f1)

        f2, a2 = plt.subplots(figsize=(5, 3)); f2.patch.set_facecolor("white")
        a2.plot(t, window_vals); a2.set_title(f"{wlabel} Window")
        a2.set_xlabel("Time (s)"); a2.set_ylabel("Amplitude"); a2.grid(True, alpha=0.3)
        figs.append(f2)

        f3, a3 = plt.subplots(figsize=(5, 3)); f3.patch.set_facecolor("white")
        a3.plot(w, 20*np.log10(np.abs(h_filter)+1e-10))
        a3.set_title(f"{ftype} Filter Frequency Response")
        a3.set_xlabel("Frequency (Hz)"); a3.set_ylabel("Magnitude (dB)")
        a3.set_ylim([-80, 5]); a3.grid(True, alpha=0.3)
        if ftype == "Lowpass": a3.axvline(x=highcut, linestyle="--", alpha=0.7)
        elif ftype == "Highpass": a3.axvline(x=lowcut, linestyle="--", alpha=0.7)
        else: a3.axvline(x=lowcut, linestyle="--", alpha=0.7); a3.axvline(x=highcut, linestyle="--", alpha=0.7)
        if show_phase:
            a3b = a3.twinx(); a3b.plot(w, np.unwrap(np.angle(h_filter)), linestyle="--", alpha=0.6)
            a3b.set_ylabel("Phase (rad)")
        figs.append(f3)

        f4, a4 = plt.subplots(figsize=(5, 3)); f4.patch.set_facecolor("white")
        a4.plot(w_win, 20*np.log10(np.abs(h_window)+1e-10))
        a4.set_title("Window Function Frequency Response")
        a4.set_xlabel("Frequency (Hz)"); a4.set_ylabel("Magnitude (dB)")
        a4.set_ylim([-100, 10]); a4.grid(True, alpha=0.3)
        figs.append(f4)

        f5, a5 = plt.subplots(figsize=(5, 3)); f5.patch.set_facecolor("white")
        a5.plot(tt, x, linewidth=1, label="Input", alpha=0.8)
        a5.plot(tt, y, linewidth=1.3, label="Filtered")
        a5.set_title("Example Signal: Raw vs Filtered")
        a5.set_xlabel("Time (s)"); a5.set_ylabel("Amplitude")
        a5.grid(True, alpha=0.3); a5.legend()
        figs.append(f5)

        render_to_imgs(figs)
        current = {"taps": taps, "fs": fs, "numtaps": numtaps, "filter_type": ftype,
                   "window_type": wlabel, "lowcut": lowcut, "highcut": highcut, "figs": figs}
        document.getElementById("downloadCSV").disabled = False
        document.getElementById("downloadPlot").disabled = False
        set_status(f"Filter: {ftype}, {numtaps} taps, Window: {wlabel}")
    except Exception as e:
        set_status(f"Error: {e}")
        console.log(f"Design error: {e}")

def export_csv(event=None) -> None:
    if not current:
        set_status("No filter yet. Design first.")
        return
    lines = ["index,coefficient"] + [f"{i},{c}" for i, c in enumerate(current["taps"])]
    download_bytes("filter_coefficients.csv", ("\n".join(lines)+"\n").encode("utf-8"), "text/csv")
    meta = (f"Filter Type: {current['filter_type']}\n"
            f"Window Type: {current['window_type']}\n"
            f"Sampling Frequency: {current['fs']} Hz\n"
            f"Filter Length: {current['numtaps']}\n"
            f"Low Cutoff: {current['lowcut']} Hz\n"
            f"High Cutoff: {current['highcut']} Hz\n").encode("utf-8")
    download_bytes("filter_meta.txt", meta, "text/plain")
    set_status("CSV files exported.")

def export_plots(event=None) -> None:
    if not current:
        set_status("No plots yet. Design first.")
        return
    if not PIL_AVAILABLE:
        for i, fig in enumerate(current["figs"], start=1):
            buf = BytesIO(); fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white"); buf.seek(0)
            download_bytes(f"fir_plot_{i}.png", buf.getvalue(), "image/png")
        set_status("Exported individual plot PNGs."); return
    images = []
    for fig in current["figs"]:
        buf = BytesIO(); fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white"); buf.seek(0)
        images.append(Image.open(buf).convert("RGBA"))
    max_w = max(im.width for im in images); total_h = sum(im.height for im in images)
    out = Image.new("RGBA", (max_w, total_h), (255, 255, 255, 255))
    y = 0
    for im in images:
        x_off = (max_w - im.width) // 2; out.paste(im, (x_off, y)); y += im.height
    out_buf = BytesIO(); out.save(out_buf, format="PNG"); out_buf.seek(0)
    download_bytes("fir_plots.png", out_buf.getvalue(), "image/png")
    set_status("Exported combined plot PNG.")

async def initialize_app():
    global _EVENTS_BOUND, _PROXIES
    try:
        wsel = document.getElementById("wtype")
        if wsel.length == 0:
            for k in window_map.keys():
                opt = document.createElement("option"); opt.text = k; opt.value = k; wsel.add(opt)
            wsel.value = "Hamming"
    except Exception as e:
        console.log(f"Window select init error: {e}")

    if not _EVENTS_BOUND:
        _PROXIES["design"] = create_proxy(design)
        _PROXIES["reset"]  = create_proxy(reset_defaults)
        _PROXIES["csv"]    = create_proxy(export_csv)
        _PROXIES["plot"]   = create_proxy(export_plots)
        _PROXIES["wchg"]   = create_proxy(update_tooltip)

        document.getElementById("designBtn").addEventListener("click", _PROXIES["design"])
        document.getElementById("resetBtn").addEventListener("click", _PROXIES["reset"])
        document.getElementById("downloadCSV").addEventListener("click", _PROXIES["csv"])
        document.getElementById("downloadPlot").addEventListener("click", _PROXIES["plot"])
        document.getElementById("wtype").addEventListener("change", _PROXIES["wchg"])
        _EVENTS_BOUND = True

    reset_defaults()
    design()

asyncio.create_task(initialize_app())
