from __future__ import annotations
import argparse, logging, sys, time, traceback
from pathlib import Path

# ── bootstrap: ensure cwd is the script folder ──────────────────────────────
import os
os.chdir(Path(__file__).parent)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent / "src"))

import config
from module_runner import load

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(module)s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(config.OUTPUT_DIR) / "pipeline.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


def _parse_args():
    p = argparse.ArgumentParser(description="Geodata Processing AI Pipeline")
    p.add_argument("--tif", nargs="*", default=[], metavar="NAME=PATH",
                   help="Extra TIF bands  e.g.  B6_SWIR2=/data/b6.tif")
    p.add_argument("--csv", nargs="*", default=[], metavar="KEY=PATH",
                   help="Override CSV: key = 'water' or 'vizag'")
    p.add_argument("--before-tif", nargs="*", default=[], metavar="NAME=PATH",
                   help="T1 bands for real two-date change detection")
    p.add_argument("--no-viz", action="store_true", help="Skip visualisation")
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
             "  GEODATA PROCESSING AI PIPELINE – SUMMARY REPORT",
             sep,
             f"  Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
             ""]

    def section(title, items):
        lines.extend([sep, f"  {title}", sep])
        for k, v in items:
            lines.append(f"  {k:<22}: {v}")
        lines.append("")

    bands_shape = list(ingest.bands.values())[0].shape
    section("MODULE 1 – DATA INGESTION", [
        ("Bands loaded",        list(ingest.bands.keys())),
        ("CRS",                 ingest.crs),
        ("Grid size",           bands_shape),
        ("water_train rows",    ingest.df_water.shape[0]),
        ("vizag rows",          ingest.df_vizag.shape[0]),
        ("Runtime",             f"{timing['ingestion']:.1f}s"),
    ])

    qa = preproc.qa_stats
    section("MODULE 2 – PRE-PROCESSING & QA", [
        ("Cloud pixels",        f"{qa['cloud_pct']:.1f}%"),
        ("Shadow pixels",       f"{qa['shadow_pct']:.1f}%"),
        ("Valid pixels",        f"{qa['valid_pct']:.1f}%"),
        ("Mean QA confidence",  f"{qa['mean_qa']:.3f}"),
        ("Indices computed",    list(preproc.indices.keys())),
        ("Runtime",             f"{timing['preprocessing']:.1f}s"),
    ])

    section("MODULE 3 – BIAS MITIGATION", [
        ("water_train CV",      harmon.bias_report["water"]["cv"]),
        ("vizag CV",            harmon.bias_report["vizag"]["cv"]),
        ("Bias reduced",        harmon.bias_report["summary"]["bias_reduced"]),
        ("Runtime",             f"{timing['harmonisation']:.1f}s"),
    ])

    model_lines = []
    for name, m in model_result.metrics.items():
        model_lines.append((name,
            f"Acc={m['accuracy']:.4f}  F1(macro)={m['f1_macro']:.4f}  F1(wt)={m['f1_weighted']:.4f}"))
    model_lines.append(("Runtime", f"{timing['ai_models']:.1f}s"))
    section("MODULE 4 – AI MODELS", model_lines)

    cs = change_result.stats
    section("MODULE 5 – CHANGE DETECTION", [
        ("Total pixels",        f"{cs['total_pixels']:,}"),
        ("Changed pixels",      f"{cs['changed_pixels']:,}  ({cs['change_pct']:.2f}%)"),
        ("Vegetation loss px",  f"{cs['vegetation_loss_px']:,}"),
        ("Urban expansion px",  f"{cs['urban_expansion_px']:,}"),
        ("Water change px",     f"{cs['water_change_px']:,}"),
        ("Anomaly px",          f"{cs['anomaly_px']:,}"),
        ("Runtime",             f"{timing['change_detection']:.1f}s"),
    ])

    section("MODULE 6 – VISUALISATION", [
        ("Figures dir",         config.OUT["viz_dir"]),
        ("Runtime",             f"{timing.get('visualization', 0):.1f}s"),
    ])

    total = sum(timing.values())
    lines.extend([sep, "  PIPELINE COMPLETE", sep,
                  f"  Total runtime  : {total:.1f}s",
                  f"  Output dir     : {config.OUTPUT_DIR}",
                  f"  Water model    : {config.OUT['water_model']}",
                  f"  LC model       : {config.OUT['lc_model']}",
                  f"  Change map     : {config.OUT['change_map_tif']}",
                  f"  Water map pred : {config.OUT['water_map_pred_tif']}",
                  f"  LC map         : {config.OUT['lc_map_tif']}",
                  sep, ""])

    text = "\n".join(lines)
    print(text)
    with open(config.OUT["report_txt"], "w") as f:
        f.write(text)
    log.info("Report written → %s", config.OUT["report_txt"])


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = _parse_args()
    extra_tifs       = _kv(args.tif)
    extra_csvs       = _kv(args.csv)
    before_tif_paths = _kv(args.before_tif)

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   GEODATA PROCESSING AI PIPELINE  ─  START              ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    timing = {}

    # ── 1. Ingestion ─────────────────────────────────────────────────────────
    t0 = time.time()
    m1 = load("01_data_ingestion")
    ingest = m1.run_ingestion(
        user_tif_paths=extra_tifs or None,
        user_csv_paths=extra_csvs or None,
    )
    timing["ingestion"] = time.time() - t0

    # ── 2. Pre-processing ─────────────────────────────────────────────────────
    t0 = time.time()
    m2 = load("02_preprocessing")
    preproc = m2.run_preprocessing(ingest)
    timing["preprocessing"] = time.time() - t0

    # ── 3. Harmonisation ──────────────────────────────────────────────────────
    t0 = time.time()
    m3 = load("03_bias_harmonization")
    harmon = m3.run_harmonisation(ingest, preproc)
    timing["harmonisation"] = time.time() - t0

    # ── 4. AI Models ──────────────────────────────────────────────────────────
    t0 = time.time()
    m4 = load("04_ai_models")
    model_result = m4.run_ai_models(ingest, preproc, harmon)
    timing["ai_models"] = time.time() - t0

    # ── 5. Change Detection ───────────────────────────────────────────────────
    t0 = time.time()
    before_indices = None
    if before_tif_paths:
        log.info("Loading user 'before' bands for genuine two-date CD …")
        import rasterio, numpy as np
        before_raw = {}
        for name, path in before_tif_paths.items():
            with rasterio.open(path) as src:
                before_raw[name] = src.read(1).astype(np.float32) * config.DN_SCALE_FACTOR
        if len(before_raw) >= 4:
            keys = list(before_raw.keys())
            G_b  = before_raw.get("B2_Green", before_raw[keys[0]])
            R_b  = before_raw.get("B3_Red",   before_raw[keys[1]])
            N_b  = before_raw.get("B4_NIR",   before_raw[keys[2]])
            S_b  = before_raw.get("B5_SWIR",  before_raw[keys[3]])
            sr   = lambda a, b: np.clip((a-b)/(a+b+1e-6), -1, 1)
            before_indices = {
                "NDVI":  sr(N_b, R_b), "NDWI":  sr(G_b, N_b),
                "MNDWI": sr(G_b, S_b), "NDBI":  sr(S_b, N_b),
                "NBR":   sr(N_b, S_b),
            }

    m5 = load("05_change_detection")
    change_result = m5.run_change_detection(preproc, model_result,
                                            before_indices=before_indices)
    timing["change_detection"] = time.time() - t0

    # ── 6. Visualisation ──────────────────────────────────────────────────────
    if not args.no_viz:
        t0 = time.time()
        m6 = load("06_visualization")
        m6.run_visualization(ingest, preproc, harmon, model_result, change_result)
        timing["visualization"] = time.time() - t0
    else:
        log.info("Visualisation skipped (--no-viz).")

    _write_report(timing, ingest, preproc, harmon, model_result, change_result)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("Pipeline failed:\n%s", traceback.format_exc())
        sys.exit(1)
