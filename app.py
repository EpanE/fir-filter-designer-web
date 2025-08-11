from js import document, URL, Blob

    except Exception as e:
        set_status(f"Error: {e}")


def export_csv(event=None):
    if not current:
        set_status('No filter yet. Design first.'); return
    import csv
    from io import StringIO
    sio = StringIO()
    writer = csv.writer(sio)
    writer.writerow(['index','coefficient'])
    for i, c in enumerate(current['taps']):
        writer.writerow([i, c])
    meta = (
        f"Filter Type: {current['filter_type']}\n"
        f"Window Type: {current['window_type']}\n"
        f"Sampling Frequency: {current['fs']} Hz\n"
        f"Filter Length: {current['numtaps']}\n"
        f"Low Cutoff: {current['lowcut']} Hz\n"
        f"High Cutoff: {current['highcut']} Hz\n"
    )
    download_bytes('filter_coefficients.csv', sio.getvalue().encode('utf-8'), 'text/csv')
    download_bytes('filter_meta.txt', meta.encode('utf-8'), 'text/plain')


def export_plots(event=None):
    if not current:
        set_status('No plots yet. Design first.'); return
    # Stack all figures vertically into one PNG
    imgs = []
    for fig in current['figs']:
        buf = BytesIO(); fig.savefig(buf, dpi=140, bbox_inches='tight'); buf.seek(0)
        imgs.append(Image.open(buf).convert('RGBA'))
    widths = [im.width for im in imgs]
    max_w = max(widths)
    total_h = sum(im.height for im in imgs)
    out = Image.new('RGBA', (max_w, total_h), (0,0,0,0))
    y=0
    for im in imgs:
        if im.width < max_w:
            im = ImageOps.expand(im, border=(0,0,max_w-im.width,0), fill=(14,22,38,255))
        out.paste(im, (0,y)); y += im.height
    bio = BytesIO(); out.save(bio, format='PNG'); bio.seek(0)
    download_bytes('fir_plots.png', bio.getvalue(), 'image/png')

# Hook buttons
_document = document
_document.getElementById('designBtn').addEventListener('click', design)
_document.getElementById('resetBtn').addEventListener('click', reset_defaults)
_document.getElementById('downloadCSV').addEventListener('click', export_csv)
_document.getElementById('downloadPlot').addEventListener('click', export_plots)

# Initial defaults
reset_defaults()
