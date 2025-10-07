# streamlit_app.py
# Streamlit wrapper for the NSS BMP analyzer

import io
import os
import zipfile
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import scipy.signal
import py7zr

# ===========================
# Original helper functions
# ===========================

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def find_nonblack(npy1d, threshold):
    # Fall back to simple threshold if peak finder returns nothing
    try:
        # find_peaks_cwt can be finicky; use a fixed width list for stability
        peaks = scipy.signal.find_peaks_cwt(npy1d, widths=np.array([10]))
    except Exception:
        peaks = []
    res = None
    # First peak above threshold
    for p in peaks:
        if npy1d[p] >= threshold:
            res = p
            break
    # Fallback: first sample above threshold
    if res is None:
        idx = np.argmax(npy1d >= threshold)
        res = int(idx) if npy1d[idx] >= threshold else 0
    return res

def convet_nss_rawimage(img_file: str) -> np.ndarray:
    """
    Reads raw NSS BMP -> unwraps into 360*1038 columns (rotated + LR flipped)
    Returns empty array on failure.
    """
    img_v = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    if img_v is None:
        st.warning(f"Failed to read image: {img_file}")
        return np.array([])
    if img_v.shape[0] == 384000 and img_v.shape[1] == 512:
        col_u = img_v[:, 5]    # upper notch column
        col_l = img_v[:, 507]  # lower notch column
        res_u = 0
        res_l = 0
        for x, val in enumerate(col_u):
            if val >= 254:
                res_u = x; break
        for x, val in enumerate(col_l):
            if val >= 254:
                res_l = x; break
        res = np.max([res_u, res_l])
        if res > 0:
            end_1 = int((img_v.shape[0] - res) / 1038) - 2
            top_1 = (359 - end_1 + 1)
            img_1 = img_v[res:res + end_1 * 1038, :]
            img_2 = img_v[res - top_1 * 1038:res, :]
            img_v1 = cv2.vconcat([img_1, img_2])
            for i in range(0, 360):
                cv2.line(img_v1, (img_v1.shape[1] - 45, i * 1038), (img_v1.shape[1], i * 1038), (255, 255, 255), 3)
                cv2.putText(img_v1, str(i), (img_v1.shape[1] - 60, i * 1038 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            img_v1 = cv2.rotate(img_v1, cv2.ROTATE_90_CLOCKWISE)
            img_v1 = cv2.flip(img_v1, 1)  # flip left-right
            return img_v1
        else:
            st.error("Can't find the notch.")
            return np.array([])
    else:
        st.error(f"Invalid Image shape {img_v.shape}; expected (384000, 512).")
        return np.array([])

def turning_points(array):
    """Return idx_min, idx_max for local turning points."""
    idx_max, idx_min = [], []
    if (len(array) < 3):
        return idx_min, idx_max
    NEUTRAL, RISING, FALLING = range(3)

    def get_state(a, b):
        if a < b: return RISING
        if a > b: return FALLING
        return NEUTRAL

    ps = get_state(array[0], array[1])
    begin = 1
    for i in range(2, len(array)):
        s = get_state(array[i - 1], array[i])
        if s != NEUTRAL:
            if ps != NEUTRAL and ps != s:
                if s == FALLING:
                    idx_max.append((begin + i - 1) // 2)
                else:
                    idx_min.append((begin + i - 1) // 2)
            begin = i
            ps = s
    return idx_min, idx_max

# ===========================
# Lightly refactored analyzers
# ===========================

def process_bmp(bmpfile: str, out_dir: Path, mv_sdev_window_1: int = 1038) -> List:
    """
    Newer path: computes raw-range and moving-window variance channels.
    Saves:
      - *_360chart.jpg
      - *.png (annotated full wafer)
      - a folder of cropped worst-angle images
    Returns summary list for DataFrame.
    """
    f = bmpfile
    nss_img_path = Path(bmpfile).parent
    img_file = bmpfile
    img_v = convet_nss_rawimage(img_file)
    if img_v.size == 0:
        return []

    if img_v.shape[1] < (360 * mv_sdev_window_1):
        st.warning('Invalid NSS bmp file (unexpected width).')
        return []
    # Detect wafer position using multiple columns
    cols = [5000, 5500, 10000, 10500, 20000, 21000]
    th_ = 50
    res_candidates = []
    for c in cols:
        c = min(c, img_v.shape[1]-1)
        res_candidates.append(find_nonblack(img_v[:, c], th_))
    res = int(np.min(res_candidates))
    c1 = res + 10
    c2 = res + 230

    # Column stat signals
    # raw range channel
    a0_0 = np.max(img_v[c1:c2, :], axis=0) - np.min(img_v[c1:c2, :], axis=0)
    a0_1 = np.std(img_v[c1:c2, :], axis=0)  # base for moving window
    x = np.arange(0, a0_1.shape[0])

    # Normalize raw range to local floor in central region
    safe_lo, safe_hi = mv_sdev_window_1 * 3, mv_sdev_window_1 * -3
    floor = np.min(a0_0[safe_lo:safe_hi]) if a0_0.shape[0] > (abs(safe_hi) + safe_lo) else np.min(a0_0)
    a01_0 = a0_0 - floor
    a01_1 = np.std(rolling_window(a0_1, mv_sdev_window_1), 1) ** 2

    # Angle axes
    ax_0 = x / mv_sdev_window_1
    ax_1 = moving_average(x / mv_sdev_window_1, mv_sdev_window_1)

    # Central region (exclude notch/edges: 3 rev-windows on both sides)
    lo = mv_sdev_window_1 * 3
    hi = -mv_sdev_window_1 * 3 if a01_1.shape[0] > (mv_sdev_window_1 * 6) else -1
    a02_0 = a01_0[lo:hi]
    a02_1 = a01_1[lo:hi]

    Ra_0 = float(np.sum(np.abs(a02_0)) / max(1, a02_0.shape[0]))
    Q50_0 = float(np.percentile(a02_0, 50)) if a02_0.size else 0.0
    Q90_0 = float(np.percentile(a02_0, 90)) if a02_0.size else 0.0
    Q99_0 = float(np.percentile(a02_0, 99)) if a02_0.size else 0.0

    Ra_1 = float(np.sum(np.abs(a02_1)) / max(1, a02_1.shape[0]))
    Q50_1 = float(np.percentile(a02_1, 50)) if a02_1.size else 0.0
    Q90_1 = float(np.percentile(a02_1, 90)) if a02_1.size else 0.0
    Q99_1 = float(np.percentile(a02_1, 99)) if a02_1.size else 0.0

    # Per-degree metrics (3..357 inclusive)
    rpt_360 = []
    for i in range(3, 358):
        sl0 = slice(mv_sdev_window_1 * i, mv_sdev_window_1 * (i + 1))
        a_0 = a01_0[sl0]
        a_1 = a01_1[sl0 if (mv_sdev_window_1 * (i + 1)) <= a01_1.shape[0] else slice(0, 0)]
        a_0_98 = np.percentile(a_0, 98) if a_0.size else 0.0
        ra_0 = float(np.mean(a_0_98))
        ra_1 = float(np.sum(np.abs(a_1)) / max(1, a_1.shape[0])) if a_1.size else 0.0
        rpt_360.append([i, ra_0, ra_1])

    a_360 = np.array(rpt_360) if rpt_360 else np.zeros((0, 3))
    a_360_x = a_360[:, 0] if a_360.size else np.array([])
    a_360_y0 = a_360[:, 1] if a_360.size else np.array([])
    a_360_y1 = a_360[:, 2] if a_360.size else np.array([])

    # Peaks
    _, idx_max0 = turning_points(a_360_y0.tolist() if a_360_y0.size else [])
    _, idx_max1 = turning_points(a_360_y1.tolist() if a_360_y1.size else [])
    rpt_360_peaks0 = [[a_360_x[i], a_360_y0[i]] for i in idx_max0] if idx_max0 else []
    rpt_360_peaks1 = [[a_360_x[i], a_360_y1[i]] for i in idx_max1] if idx_max1 else []

    a_360_peaks0 = np.array(sorted(rpt_360_peaks0, key=lambda r: r[1])) if rpt_360_peaks0 else np.zeros((0,2))
    a_360_peaks1 = np.array(sorted(rpt_360_peaks1, key=lambda r: r[1])) if rpt_360_peaks1 else np.zeros((0,2))

    basename = Path(img_file).stem

    # 360 chart
    plt.figure(figsize=(8, 4))
    plt.title('NSS EDGE Image Quality Ra by Deg')
    plt.plot(a_360_x, a_360_y0, label=f'{basename} (raw)')
    plt.plot(a_360_x, a_360_y1, label=f'{basename} (Moving Avg/Original Ra)')
    if a_360_peaks0.shape[0] >= 10:
        plt.scatter(a_360_peaks0[-10:, 0], a_360_peaks0[-10:, 1], facecolors='none', edgecolors='r')
        plt.scatter(a_360_peaks0[-20:-10, 0], a_360_peaks0[-20:-10, 1], facecolors='none', edgecolors='g')
    if a_360_peaks1.shape[0] >= 10:
        plt.scatter(a_360_peaks1[-10:, 0], a_360_peaks1[-10:, 1], facecolors='none', edgecolors='r')
        plt.scatter(a_360_peaks1[-20:-10, 0], a_360_peaks1[-20:-10, 1], facecolors='none', edgecolors='g')
    plt.ylim(0, 150)
    plt.xticks(np.arange(0, 360, 10.0), rotation=90, ha='left')
    plt.xlabel('Angle NotchDown CW')
    plt.legend()
    chart_path = out_dir / f"{basename}_360chart.jpg"
    plt.savefig(str(chart_path), bbox_inches='tight', dpi=150)
    st.pyplot(plt.gcf())
    plt.close()

    # Overlay signals on full wafer image and save
    img_overlay = img_v.copy()
    for i in np.arange(mv_sdev_window_1 * 3, a0_1.shape[0] - mv_sdev_window_1 * 3):
        x0 = int(i)
        x1 = int(i + 1)
        y0 = int(a01_0[i] / 100 * 200)
        y1 = int(a01_0[i + 1] / 100 * 200)
        x2 = int(i) + int(mv_sdev_window_1 / 2)
        x3 = int(i + 1) + int(mv_sdev_window_1 / 2)
        y2 = int(a01_1[i] / 100 * 200) if i < len(a01_1) else 0
        y3 = int(a01_1[i + 1] / 100 * 200) if (i + 1) < len(a01_1) else 0
        cv2.line(img_overlay, (x0, img_overlay.shape[0] - y0 - 10), (x1, img_overlay.shape[0] - y1 - 10), (255, 255, 255), 1)
        cv2.line(img_overlay, (x2, img_overlay.shape[0] - y2 - 10), (x3, img_overlay.shape[0] - y3 - 10), (255, 255, 255), 2)
    wafer_png = out_dir / f"{basename}.png"
    cv2.imwrite(str(wafer_png), img_overlay)

    # Crops for top 20 peaks from both series
    crop_dir = out_dir / basename
    crop_dir.mkdir(parents=True, exist_ok=True)
    def _crop_angles(peaks_arr: np.ndarray):
        seen = set()
        for angle in peaks_arr[-20:, 0] if peaks_arr.shape[0] else []:
            a = int(angle)
            if a in seen: continue
            seen.add(a)
            x0 = int((a - 0.3) * mv_sdev_window_1)
            x1 = int((a + 1.6) * mv_sdev_window_1)
            x0 = max(0, x0); x1 = min(img_overlay.shape[1], x1)
            img_v0 = img_overlay[:, x0:x1]
            cv2.imwrite(str(crop_dir / f"{a}.png"), img_v0)
    _crop_angles(a_360_peaks0)
    _crop_angles(a_360_peaks1)

    return [str(Path(img_file).name), Ra_0, Q50_0, Q90_0, Q99_0, Ra_1, Q50_1, Q90_1, Q99_1]

def extract_and_prepare(files: List[io.BytesIO], workdir: Path, do_decompress: bool) -> List[Path]:
    """
    Writes uploaded files to workdir.
    If do_decompress: extract .7z archives and rename BMPs to waferid.bmp (when name has @).
    Returns list of BMP file paths ready to analyze.
    """
    bmp_paths: List[Path] = []

    # Save uploads
    saved_paths: List[Path] = []
    for uf in files:
        p = workdir / uf.name
        with open(p, "wb") as f:
            f.write(uf.getvalue())
        saved_paths.append(p)

    if do_decompress:
        # Extract all .7z in place
        for p in saved_paths:
            if p.suffix.lower() == ".7z":
                try:
                    with py7zr.SevenZipFile(p, mode="r") as z:
                        z.extractall(path=workdir / p.stem)
                except Exception as e:
                    st.error(f"Failed to extract {p.name}: {e}")

        # Walk extracted folders and rename BMPs
        for root, _, files_in in os.walk(workdir):
            for fn in files_in:
                if fn.lower().endswith(".bmp"):
                    full = Path(root) / fn
                    # Derive waferid from nearest .7z name pattern or the bmp name itself
                    # Expected pattern: LOT@slot@datetime@300RXM06@EDL_2...
                    # We'll try to parse based on '@' in the filename; fallback to original.
                    name_for_id = fn
                    if "@" in name_for_id:
                        try:
                            position = name_for_id.index("@")
                            waferid = name_for_id[position-9:position] + name_for_id[position+1:position+3]
                        except Exception:
                            waferid = Path(fn).stem
                    else:
                        waferid = Path(fn).stem
                    newp = Path(root) / f"{waferid}.bmp"
                    # Avoid clobber
                    if newp.exists() and str(newp) != str(full):
                        # Make unique
                        k = 1
                        while True:
                            cand = Path(root) / f"{waferid}_{k}.bmp"
                            if not cand.exists():
                                newp = cand; break
                            k += 1
                    if str(newp) != str(full):
                        try:
                            full.rename(newp)
                        except Exception:
                            pass

        # Collect BMPs at any depth
        for p in workdir.rglob("*.bmp"):
            bmp_paths.append(p)
    else:
        # No decompress: use .bmp uploads directly; if .7z uploaded, ignore until user enables decompress
        for p in saved_paths:
            if p.suffix.lower() == ".bmp":
                bmp_paths.append(p)

    # De-duplicate & sort
    bmp_paths = sorted(list(dict.fromkeys(bmp_paths)))
    return bmp_paths

def pack_outputs_zip(base_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in base_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(base_dir)))
    buf.seek(0)
    return buf.read()

# ===========================
# Streamlit UI
# ===========================

st.set_page_config(page_title="NSS BMP Analyzer", layout="wide")
st.title("NSS EDGE Image Analyzer (Streamlit)")

with st.sidebar:
    st.header("Options")
    do_decompress = st.checkbox("Extract .7z and auto-rename BMPs", value=False,
                                help="Enable if you are uploading the raw NSS .7z archives.")
    mv_window = st.number_input("Window (pixels per degree)", min_value=100, max_value=2000, value=1038, step=1)
    show_crops = st.checkbox("Show cropped worst-angle snippets", value=False)
    st.caption("Note: Files are processed in an isolated temp workspace.")

uploads = st.file_uploader(
    "Upload `.7z` and/or `.bmp` files",
    type=["7z", "bmp"],
    accept_multiple_files=True
)

run_clicked = st.button("Run analysis", type="primary", disabled=(not uploads))

if run_clicked:
    if not uploads:
        st.warning("Please upload at least one .7z or .bmp file.")
        st.stop()

    with st.spinner("Preparing files..."):
        workdir = Path(tempfile.mkdtemp(prefix="nss_streamlit_"))
        out_dir = workdir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        bmp_files = extract_and_prepare(uploads, workdir, do_decompress)

    if not bmp_files:
        st.error("No BMP files found to analyze. If you uploaded .7z, enable 'Extract .7z'.")
        st.stop()

    st.success(f"Found {len(bmp_files)} BMP file(s). Starting analysis...")

    summaries = []
    preview_tabs = st.tabs([Path(b).stem for b in bmp_files])

    for tab, bmp in zip(preview_tabs, bmp_files):
        with tab:
            st.write(f"**File:** `{Path(bmp).name}`")
            try:
                row = process_bmp(str(bmp), out_dir=out_dir, mv_sdev_window_1=int(mv_window))
            except Exception as e:
                st.exception(e)
                row = []
            if row:
                summaries.append(row)
                # Show last saved wafer png and 360 chart if present
                basename = Path(bmp).stem
                chart = out_dir / f"{basename}_360chart.jpg"
                wafer_png = out_dir / f"{basename}.png"
                if wafer_png.exists():
                    st.image(str(wafer_png), caption=f"{basename}.png (annotated)", use_column_width=True)
                if show_crops:
                    crop_dir = out_dir / basename
                    if crop_dir.exists():
                        crop_imgs = sorted(crop_dir.glob("*.png"))
                        if crop_imgs:
                            st.subheader("Worst-angle crops")
                            cols = st.columns(4)
                            for i, cp in enumerate(crop_imgs):
                                cols[i % 4].image(str(cp), caption=cp.name, use_column_width=True)
                        else:
                            st.info("No crops were generated.")
            else:
                st.warning("Analysis returned no data for this file.")

    # Results table + downloads
    if summaries:
        df = pd.DataFrame(
            summaries,
            columns=['filename','Ra_raw','RawQ50','RawQ90','RawQ99','Ra_mv','MvQ50','MvQ90','MvQ99']
        )
        st.subheader("Summary")
        st.dataframe(df, use_container_width=True)

        # Excel download
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="sheet1", index=False)
        excel_buf.seek(0)
        st.download_button("Download summary Excel", data=excel_buf.read(),
                           file_name="nss_image_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # All outputs as ZIP
        zip_bytes = pack_outputs_zip(out_dir)
        st.download_button("Download all generated images (ZIP)",
                           data=zip_bytes, file_name="nss_outputs.zip", mime="application/zip")
    else:
        st.info("No summary rows produced. Check logs above for errors or file format issues.")

# Footer tips
with st.expander("Notes & tips"):
    st.markdown("""
- The analyzer expects raw NSS BMP images of shape **384000×512**; other sizes are rejected.
- If you upload `.7z` archives, enable **Extract .7z and auto-rename BMPs**.
- The default rolling window is **1038 px ≈ 1°**; adjust if your scan resolution differs.
- Peak detection is heuristic; extreme/noisy data may produce fewer crops.
- This app does **not** use `white_paper_tools.auto_fit`; column sizing is handled by Streamlit/XlsxWriter.
""")

