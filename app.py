# app.py - FIR Filter Designer Backend
# This file contains all the Python logic for the FIR filter designer

from js import document, URL, Blob, console
import asyncio
import numpy as np
from scipy.signal import firwin, freqz, get_window, lfilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    console.log("PIL not available, plot export will be limited")

# === Window mapping & tooltips ===
window_map = {
    "Hamming": "hamming",
    "Rectangular": "boxcar",
    "Triangular": "triang",
    "Bartlett": "bartlett",
    "Sine": "sine",
    "Hann": "hann",
    "Blackman": "blackman",
    "Nuttall": "nuttall",
    "BlackmanNuttall": "blackmannuttall",
    "BlackmanHarris": "blackmanharris",
    "FlatTop": "flattop"
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
    "FlatTop": "FlatTop: Very low passband ripple"
}

# Global variable to store current filter design
current = None

# === Utility Functions ===

def set_status(msg: str):
    """Update the status message displayed to the user"""
    try:
        document.getElementById('status').innerText = msg
        console.log(f"Status: {msg}")
    except Exception as e:
        console.log(f"Error setting status: {e}")

def _img_url_from_fig(fig):
    """Convert a matplotlib figure to a blob URL for display"""
    try:
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=140, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        blob = Blob.new([buf.getvalue()], {"type": "image/png"})
        return URL.createObjectURL(blob)
    except Exception as e:
        console.log(f"Error creating image URL: {e}")
        return ""

def render_to_imgs(figs):
    """Render matplotlib figures to HTML img elements"""
    try:
        ids = ['plot1','plot2','plot3','plot4','plot5']
        for i, fig in enumerate(figs):
            if i < len(ids):
                url = _img_url_from_fig(fig)
                if url:
                    document.getElementById(ids[i]).src = url
        # Show plots grid on first render
        document.getElementById('plots').style.display = 'grid'
    except Exception as e:
        console.log(f"Error rendering images: {e}")
        set_status(f"Error displaying plots: {e}")

def download_bytes(filename: str, bytes_data: bytes, mime: str):
    """Download data as a file"""
    try:
        blob = Blob.new([bytes_data], {"type": mime})
        href = URL.createObjectURL(blob)
        a = document.createElement('a')
        a.href = href
        a.download = filename
        a.click()
        URL.revokeObjectURL(href)
    except Exception as e:
        console.log(f"Error downloading file: {e}")
        set_status(f"Download error: {e}")

# === UI Initialization Functions ===

def init_window_selector():
    """Initialize the window function dropdown"""
    try:
        wsel = document.getElementById('wtype')
        # Clear existing options
        wsel.innerHTML = ""
        # Add options
        for k in window_map.keys():
            opt = document.createElement('option')
            opt.text = k
            opt.value = k
            wsel.add(opt)
        wsel.value = 'Hamming'
        update_tooltip()
    except Exception as e:
        console.log(f"Error initializing window selector: {e}")

def update_tooltip(*_):
    """Update the tooltip text for the selected window function"""
    try:
        label = document.getElementById('wtype').value
        tooltip_text = window_tooltips.get(label, 'Unknown window type')
        document.getElementById('wtip').innerText = tooltip_text
    except Exception as e:
        console.log(f"Error updating tooltip: {e}")

# === Main Control Functions ===

def reset_defaults(event=None):
    """Reset all form inputs to their default values"""
    try:
        document.getElementById('fs').value = '100'
        document.getElementById('numtaps').value = '64'
        document.getElementById('shift').value = '0'
        document.getElementById('lowcut').value = '10'
        document.getElementById('highcut').value = '20'
        document.getElementById('ftype').value = 'Lowpass'
        document.getElementById('wtype').value = 'Hamming'
        document.getElementById('response').value = 'Impulse'
        document.getElementById('showPhase').checked = False
        update_tooltip()
        set_status('Reset to defaults. Ready.')
    except Exception as e:
        console.log(f"Error resetting defaults: {e}")
        set_status(f"Reset error: {e}")

def design(event=None):
    """Main function to design the FIR filter and generate plots"""
    global current
    try:
        set_status("Designing filter...")
        console.log("Starting filter design")
        
        # Get parameters from form
        fs = float(document.getElementById('fs').value)
        numtaps = int(document.getElementById('numtaps').value)
        shift = int(document.getElementById('shift').value)
        lowcut = float(document.getElementById('lowcut').value)
        highcut = float(document.getElementById('highcut').value)
        ftype = document.getElementById('ftype').value
        wlabel = document.getElementById('wtype').value
        wtype = window_map.get(wlabel, 'hamming')
        response = document.getElementById('response').value
        show_phase = bool(document.getElementById('showPhase').checked)

        console.log(f"Parameters: fs={fs}, numtaps={numtaps}, ftype={ftype}, wtype={wtype}")

        # Parameter validation
        if fs <= 0:
            raise ValueError('Sampling frequency must be positive.')
        if numtaps < 2:
            raise ValueError('Filter length should be 2 or greater.')
        if lowcut <= 0 or highcut <= 0:
            raise ValueError('Frequencies must be greater than 0 Hz.')
        if highcut >= fs/2 or lowcut >= fs/2:
            raise ValueError(f'Cutoffs must be < Nyquist ({fs/2} Hz).')
        if (ftype in ['Bandpass','Bandstop']) and (lowcut >= highcut):
            raise ValueError('Low frequency must be less than high frequency for band filters.')

        # Design filter coefficients
        console.log("Designing filter coefficients...")
        if ftype == 'Lowpass':
            taps = firwin(numtaps, cutoff=highcut, window=wtype, fs=fs, pass_zero='lowpass')
        elif ftype == 'Highpass':
            taps = firwin(numtaps, cutoff=lowcut, window=wtype, fs=fs, pass_zero='highpass')
        elif ftype == 'Bandpass':
            taps = firwin(numtaps, cutoff=[lowcut, highcut], window=wtype, fs=fs, pass_zero=False)
        elif ftype == 'Bandstop':
            taps = firwin(numtaps, cutoff=[lowcut, highcut], window=wtype, fs=fs, pass_zero=True)
        else:
            raise ValueError('Invalid filter type')

        # Apply frequency shift if specified
        if shift != 0:
            n = np.arange(numtaps)
            taps = (taps * np.exp(1j * 2 * np.pi * shift * n / numtaps)).real

        console.log("Getting window values...")
        window_vals = get_window(wtype, numtaps, fftbins=False)

        # Time vectors
        t = np.arange(numtaps)/fs

        # Step response calculation
        console.log("Computing step response...")
        step_input = np.ones(numtaps*3)
        step_response = lfilter(taps, [1.0], step_input)
        step_time = np.arange(len(step_response))/fs

        # Frequency responses
        console.log("Computing frequency responses...")
        w, h_filter = freqz(taps, worN=8192, fs=fs)
        w_win, h_window = freqz(window_vals, worN=8192, fs=fs)

        # Generate example signal for demonstration
        console.log("Generating example signal...")
        dur = 2.0
        N = int(fs*dur)
        tt = np.arange(N)/fs
        np.random.seed(0)  # For reproducible results
        x = 0.9*np.sin(2*np.pi*5*tt) + 0.4*np.sin(2*np.pi*25*tt) + 0.05*np.random.normal(size=N)
        y = lfilter(taps, [1.0], x)

        # Create all plots
        console.log("Creating plots...")
        plt.style.use('default')  # Ensure consistent style
        figs = []
        
        # Plot 1: Impulse/Step Response
        fig1, ax1 = plt.subplots(figsize=(5,3))
        fig1.patch.set_facecolor('white')
        if response in ['Impulse','Both']:
            ax1.stem(t, taps, basefmt=" ")
            ax1.set_title('Impulse Response')
        elif response == 'Step':
            ax1.plot(step_time[:numtaps*2], step_response[:numtaps*2])
            ax1.set_title('Step Response')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        figs.append(fig1)

        # Plot 2: Window Function (time domain)
        fig2, ax2 = plt.subplots(figsize=(5,3))
        fig2.patch.set_facecolor('white')
        ax2.plot(t, window_vals)
        ax2.set_title(f'{wlabel} Window')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        figs.append(fig2)

        # Plot 3: Filter Frequency Response
        fig3, ax3 = plt.subplots(figsize=(5,3))
        fig3.patch.set_facecolor('white')
        magnitude_db = 20*np.log10(np.abs(h_filter) + 1e-10)
        ax3.plot(w, magnitude_db)
        ax3.set_title(f'{ftype} Filter Frequency Response')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Magnitude (dB)')
        ax3.set_ylim([-80,5])
        ax3.grid(True, alpha=0.3)
        
        # Add cutoff frequency markers
        if ftype == 'Lowpass':
            ax3.axvline(x=highcut, color='red', linestyle='--', alpha=0.7)
        elif ftype == 'Highpass':
            ax3.axvline(x=lowcut, color='red', linestyle='--', alpha=0.7)
        elif ftype in ['Bandpass','Bandstop']:
            ax3.axvline(x=lowcut, color='red', linestyle='--', alpha=0.7)
            ax3.axvline(x=highcut, color='red', linestyle='--', alpha=0.7)
            
        # Add phase response if requested
        if show_phase:
            ax3b = ax3.twinx()
            phase = np.unwrap(np.angle(h_filter))
            ax3b.plot(w, phase, color='orange', linestyle='--', alpha=0.7)
            ax3b.set_ylabel('Phase (rad)', color='orange')
        figs.append(fig3)

        # Plot 4: Window Frequency Response
        fig4, ax4 = plt.subplots(figsize=(5,3))
        fig4.patch.set_facecolor('white')
        window_magnitude_db = 20*np.log10(np.abs(h_window) + 1e-10)
        ax4.plot(w_win, window_magnitude_db)
        ax4.set_title('Window Function Frequency Response')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Magnitude (dB)')
        ax4.set_ylim([-100,10])
        ax4.grid(True, alpha=0.3)
        figs.append(fig4)

        # Plot 5: Example Signal (Raw vs Filtered)
        fig5, ax5 = plt.subplots(figsize=(5,3))
        fig5.patch.set_facecolor('white')
        ax5.plot(tt, x, linewidth=1, label='Input', alpha=0.8)
        ax5.plot(tt, y, linewidth=1.5, label='Filtered')
        ax5.set_title('Example Signal: Raw vs Filtered')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Amplitude')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        figs.append(fig5)

        console.log("Rendering plots...")
        render_to_imgs(figs)

        # Store current design for export functions
        current = {
            'taps': taps,
            'fs': fs,
            'numtaps': numtaps,
            'filter_type': ftype,
            'window_type': wlabel,
            'lowcut': lowcut,
            'highcut': highcut,
            'figs': figs
        }
        
        # Enable download buttons
        document.getElementById('downloadCSV').disabled = False
        document.getElementById('downloadPlot').disabled = False
        set_status(f"Filter: {ftype}, {numtaps} taps, Window: {wlabel}")
        console.log("Filter design completed successfully")

    except Exception as e:
        console.log(f"Error in design function: {e}")
        set_status(f"Error: {e}")

def export_csv(event=None):
    """Export filter coefficients and metadata as CSV files"""
    try:
        if not current:
            set_status('No filter yet. Design first.')
            return
        
        console.log("Exporting CSV...")
        
        # Create CSV content for coefficients
        csv_content = "index,coefficient\n"
        for i, c in enumerate(current['taps']):
            csv_content += f"{i},{c}\n"
        
        # Create metadata file
        meta = (
            f"Filter Type: {current['filter_type']}\n"
            f"Window Type: {current['window_type']}\n"
            f"Sampling Frequency: {current['fs']} Hz\n"
            f"Filter Length: {current['numtaps']}\n"
            f"Low Cutoff: {current['lowcut']} Hz\n"
            f"High Cutoff: {current['highcut']} Hz\n"
        )
        
        # Download both files
        download_bytes('filter_coefficients.csv', csv_content.encode('utf-8'), 'text/csv')
        download_bytes('filter_meta.txt', meta.encode('utf-8'), 'text/plain')
        set_status("CSV files exported successfully")
        
    except Exception as e:
        console.log(f"Error exporting CSV: {e}")
        set_status(f"CSV export error: {e}")

def export_plots(event=None):
    """Export plots as PNG files"""
    try:
        if not current:
            set_status('No plots yet. Design first.')
            return
        
        console.log("Exporting plots...")
        
        if PIL_AVAILABLE:
            # Stack all figures vertically into one PNG
            imgs = []
            for fig in current['figs']:
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=140, bbox_inches='tight', facecolor='white')
                buf.seek(0)
                imgs.append(Image.open(buf).convert('RGBA'))
            
            widths = [im.width for im in imgs]
            max_w = max(widths)
            total_h = sum(im.height for im in imgs)
            out = Image.new('RGBA', (max_w, total_h), (255,255,255,255))
            
            y = 0
            for im in imgs:
                if im.width < max_w:
                    # Center the image
                    x_offset = (max_w - im.width) // 2
                    out.paste(im, (x_offset, y))
                else:
                    out.paste(im, (0, y))
                y += im.height
            
            bio = BytesIO()
            out.save(bio, format='PNG')
            bio.seek(0)
            download_bytes('fir_plots.png', bio.getvalue(), 'image/png')
            set_status("Plot exported successfully")
        else:
            # Fallback: export individual plots
            for i, fig in enumerate(current['figs']):
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=140, bbox_inches='tight', facecolor='white')
                buf.seek(0)
                download_bytes(f'fir_plot_{i+1}.png', buf.getvalue(), 'image/png')
            set_status("Individual plots exported successfully")
        
    except Exception as e:
        console.log(f"Error exporting plots: {e}")
        set_status(f"Plot export error: {e}")

# === Application Initialization ===

async def initialize_app():
    """Initialize the application with event listeners and default values"""
    try:
        console.log("Initializing application...")
        
        # Initialize window selector
        init_window_selector()
        
        # Hook up event listeners
        document.getElementById('designBtn').addEventListener('click', design)
        document.getElementById('resetBtn').addEventListener('click', reset_defaults)
        document.getElementById('downloadCSV').addEventListener('click', export_csv)
        document.getElementById('downloadPlot').addEventListener('click', export_plots)
        document.getElementById('wtype').addEventListener('change', update_tooltip)
        
        # Set initial defaults
        reset_defaults()
        
        console.log("Application initialized successfully")
        set_status("Ready. Set parameters and click 'Design Filter'.")
        
    except Exception as e:
        console.log(f"Error initializing application: {e}")
        set_status(f"Initialization error: {e}")

# Start the application when the module is loaded
asyncio.create_task(initialize_app())
