"""
run_pipeline_dl.py
CNN + LSTM + Transformer

Usage
-----
  python run_pipeline_dl.py             # full DL pipeline
  python run_pipeline_dl.py --no-viz    # skip figures
  python run_pipeline_dl.py --tif B6=/path/b6.tif

This runner is identical to run_pipeline.py EXCEPT it uses
04_ai_models_dl.py instead of 04_ai_models.py.
"""

from __future__ import annotations
import argparse, logging, os, sys, time, traceback
from pathlib import Path

os.chdir(Path(__file__).parent)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent / "src"))
import config
from module_runner import load

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(module)s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(config.OUTPUT_DIR) / "pipeline_dl.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


def _parse_args():
    p = argparse.ArgumentParser(description="DL Geodata Pipeline (CNN+LSTM+Transformer)")
    p.add_argument("--tif", nargs="*", default=[], metavar="NAME=PATH")
    p.add_argument("--csv", nargs="*", default=[], metavar="KEY=PATH")
    p.add_argument("--before-tif", nargs="*", default=[], metavar="NAME=PATH")
    p.add_argument("--no-viz", action="store_true")
    return p.parse_args()


def _kv(lst):
    out = {}
    for item in (lst or []):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _write_report(timing, ingest, preproc, harmon, model_result, change_result):
    sep = "═" * 65
    lines = ["", sep,
             "  GEODATA PROCESSING AI PIPELINE (DL VERSION) – REPORT",
             "  Architecture: CNN (U-Net) + LSTM/GRU + Transformer",
             sep,
             f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]

    lines.extend([sep, "  MODULE 4 — DL MODEL PERFORMANCE", sep])
    for name, m in model_result.metrics.items():
        lines.append(f"  {name:<32}: Acc={m['accuracy']:.4f}  "
                     f"F1(macro)={m['f1_macro']:.4f}  "
                     f"F1(wt)={m['f1_weighted']:.4f}")
    lines.append("")

    cs = change_result.stats
    lines.extend([sep, "  MODULE 5 — CHANGE DETECTION", sep,
                  f"  Changed pixels : {cs['changed_pixels']:,}  ({cs['change_pct']:.2f}%)",
                  f"  Vegetation loss: {cs['vegetation_loss_px']:,}",
                  f"  Urban expansion: {cs['urban_expansion_px']:,}",
                  f"  Water change   : {cs['water_change_px']:,}", ""])

    total = sum(timing.values())
    lines.extend([sep, f"  Total runtime: {total:.1f}s",
                  f"  Output dir   : {config.OUTPUT_DIR}", sep, ""])

    text = "\n".join(lines)
    print(text)
    report_path = Path(config.OUTPUT_DIR) / "pipeline_dl_report.txt"
    with open(report_path, "w") as f:
        f.write(text)
    log.info("Report written → %s", report_path)


def main():
    args = _parse_args()
    extra_tifs  = _kv(args.tif)
    extra_csvs  = _kv(args.csv)
    before_tifs = _kv(args.before_tif)
    timing = {}

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  GEODATA PROCESSING AI PIPELINE  —  DL VERSION          ║")
    log.info("║  Models: CNN (U-Net) + LSTM/GRU + Transformer            ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    t0 = time.time()
    ingest = load("01_data_ingestion").run_ingestion(
        user_tif_paths=extra_tifs or None,
        user_csv_paths=extra_csvs or None,
    )
    timing["ingestion"] = time.time() - t0

    t0 = time.time()
    preproc = load("02_preprocessing").run_preprocessing(ingest)
    timing["preprocessing"] = time.time() - t0

    t0 = time.time()
    harmon = load("03_bias_harmonization").run_harmonisation(ingest, preproc)
    timing["harmonisation"] = time.time() - t0

    # ── MODULE 4: DL MODELS (paper-compliant) ─────────────────────────────
    t0 = time.time()
    m4_dl = load("04_ai_models_dl")
    model_result = m4_dl.run_dl_models(ingest, preproc, harmon)
    timing["dl_models"] = time.time() - t0

    # ── MODULE 5: CHANGE DETECTION ────────────────────────────────────────
    t0 = time.time()
    before_indices = None
    if before_tifs and len(before_tifs) >= 4:
        import rasterio, numpy as np
        before_raw = {}
        for name, path in before_tifs.items():
            with rasterio.open(path) as src:
                before_raw[name] = src.read(1).astype(np.float32) * config.DN_SCALE_FACTOR
        keys = list(before_raw.keys())
        G_b, R_b = before_raw.get("B2_Green", before_raw[keys[0]]), before_raw.get("B3_Red", before_raw[keys[1]])
        N_b, S_b = before_raw.get("B4_NIR",   before_raw[keys[2]]), before_raw.get("B5_SWIR", before_raw[keys[3]])
        sr = lambda a, b: np.clip((a-b)/(a+b+1e-6), -1, 1)
        before_indices = {"NDVI": sr(N_b,R_b), "NDWI": sr(G_b,N_b),
                          "MNDWI": sr(G_b,S_b), "NDBI": sr(S_b,N_b), "NBR": sr(N_b,S_b)}

    change_result = load("05_change_detection").run_change_detection(
        preproc, model_result, before_indices=before_indices
    )
    timing["change_detection"] = time.time() - t0

    # ── MODULE 6: VISUALISATION ───────────────────────────────────────────
    if not args.no_viz:
        t0 = time.time()
        load("06_visualization").run_visualization(
            ingest, preproc, harmon, model_result, change_result
        )
        timing["visualization"] = time.time() - t0

    _write_report(timing, ingest, preproc, harmon, model_result, change_result)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("DL Pipeline failed:\n%s", traceback.format_exc())
        sys.exit(1)
