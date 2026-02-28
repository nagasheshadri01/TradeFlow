"""
setup_local.py
──────────────
One-time setup script for the fully-local Sentiment Trader.

What it does
────────────
1. Downloads and saves the DistilBERT SST-2 model to  models/distilbert-sst2/
2. Downloads the NLTK VADER lexicon to the default NLTK data directory
3. Verifies both assets load correctly

Run ONCE (with internet access), then the system works entirely offline.

Usage
─────
    python setup_local.py           # download + verify everything
    python setup_local.py --check   # offline integrity check only (no download)

Optional flags
──────────────
    --model-dir  PATH   Override the local model save directory
                        (default: models/distilbert-sst2)
    --skip-model        Skip model download (only download VADER lexicon)
    --skip-vader        Skip VADER download (only download model)
    --check             Offline integrity check: verify cached files without
                        downloading anything. Exits 0 if OK, 1 if broken.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("setup_local")

# ── Project root (this file lives at the project root) ───────────────────────
ROOT = Path(__file__).resolve().parent


# ── Step 1: DistilBERT ────────────────────────────────────────────────────────

def download_model(model_id: str, save_dir: Path) -> None:
    """Download *model_id* from HuggingFace Hub and save all files to *save_dir*."""
    log.info("=" * 60)
    log.info("Downloading DistilBERT model …")
    log.info("  Source : %s", model_id)
    log.info("  Target : %s", save_dir)
    log.info("=" * 60)

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        log.error("transformers is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    save_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(str(save_dir))

    log.info("Downloading model weights …")
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.save_pretrained(str(save_dir))

    log.info("✓ Model saved to %s", save_dir)


def verify_model(save_dir: Path) -> None:
    """Smoke-test: run one inference from the local directory."""
    log.info("Verifying model loads from disk …")
    try:
        from transformers import pipeline
    except ImportError:
        log.warning("transformers not available – skipping model verification.")
        return

    pipe = pipeline(
        "sentiment-analysis",
        model=str(save_dir),
        truncation=True,
        max_length=512,
    )
    result = pipe("Strong earnings boosted investor confidence.")[0]
    log.info("  Sample result: %s", result)
    log.info("✓ Model verification passed.")


# ── Step 2: VADER Lexicon ─────────────────────────────────────────────────────

def download_vader() -> None:
    """Download the VADER lexicon into the default NLTK data directory."""
    log.info("=" * 60)
    log.info("Downloading NLTK VADER lexicon …")
    log.info("=" * 60)

    try:
        import nltk
    except ImportError:
        log.error("nltk is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    nltk.download("vader_lexicon", quiet=False)
    log.info("✓ VADER lexicon downloaded.")


def verify_vader() -> None:
    """Smoke-test: score one sentence with VADER."""
    log.info("Verifying VADER lexicon …")
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores("Great results, very happy!")["compound"]
    log.info("  Sample score: %.4f", score)
    assert score > 0, "VADER returned non-positive score for clearly positive text."
    log.info("✓ VADER verification passed.")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-time local setup for Sentiment Trader (downloads model + VADER)."
    )
    parser.add_argument(
        "--model-dir",
        default=str(ROOT / "models" / "distilbert-sst2"),
        help="Directory to save DistilBERT model weights (default: models/distilbert-sst2)",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip DistilBERT download",
    )
    parser.add_argument(
        "--skip-vader",
        action="store_true",
        help="Skip VADER lexicon download",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Offline integrity check only — verify cached model files and VADER "
            "without downloading anything. Exits 0 if everything is OK, 1 if broken."
        ),
    )
    return parser.parse_args()


# ── Integrity check (--check flag) ───────────────────────────────────────────

def check_integrity(model_dir: Path) -> bool:
    """
    Perform an offline integrity check of the local model cache and VADER lexicon.

    Checks:
    1. Model directory exists and contains config.json.
    2. At least one weight file is present:
           model.safetensors  (HuggingFace default since 2023)
           OR pytorch_model.bin  (older format)
    3. VADER lexicon is importable without a network call.
    4. A pipeline can be constructed from the local directory (dry-run).

    Returns True if all checks pass, False otherwise.
    Prints a human-readable PASS / FAIL line for each check.
    """
    ok = True
    sep = "-" * 58
    log.info(sep)
    log.info("  Offline integrity check")
    log.info("  Model dir: %s", model_dir)
    log.info(sep)

    # ── Check 1: model directory ──
    if not model_dir.exists():
        log.error("  [FAIL] Model directory does not exist: %s", model_dir)
        log.error("         Run: python setup_local.py  (to download)")
        ok = False
    else:
        log.info("  [PASS] Model directory exists.")

        # ── Check 2: config.json ──
        config_file = model_dir / "config.json"
        if not config_file.exists():
            log.error("  [FAIL] config.json missing in %s", model_dir)
            ok = False
        else:
            log.info("  [PASS] config.json found.")

        # ── Check 3: weight file ──
        weight_files = [
            model_dir / "model.safetensors",
            model_dir / "pytorch_model.bin",
        ]
        found_weights = [f for f in weight_files if f.exists()]
        if not found_weights:
            log.error(
                "  [FAIL] No weight file found in %s\n"
                "         Expected: model.safetensors or pytorch_model.bin",
                model_dir,
            )
            ok = False
        else:
            for wf in found_weights:
                size_mb = wf.stat().st_size / 1_048_576
                log.info("  [PASS] Weights: %s (%.1f MB)", wf.name, size_mb)
                if size_mb < 50:  # DistilBERT is ~255 MB; anything <50 MB is suspect
                    log.warning(
                        "  [WARN] Weight file seems very small (%.1f MB). "
                        "It may be a pointer file, not real weights. "
                        "Re-run setup to re-download.", size_mb
                    )

        # ── Check 4: model loads without network ──
        try:
            import os as _os
            # Block HuggingFace Hub outbound calls for this check
            env_backup = _os.environ.get("TRANSFORMERS_OFFLINE")
            _os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline(
                "sentiment-analysis",
                model=str(model_dir),
                truncation=True,
                max_length=64,
            )
            result = pipe("The market is performing well today.")[0]
            log.info("  [PASS] Model inference OK — label=%s score=%.4f",
                     result['label'], result['score'])
            if env_backup is None:
                _os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                _os.environ["TRANSFORMERS_OFFLINE"] = env_backup
        except Exception as exc:
            log.error("  [FAIL] Model failed to load offline: %s", exc)
            ok = False

    # ── Check 5: VADER lexicon ──
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia   = SentimentIntensityAnalyzer()
        score = sia.polarity_scores("Good results!")["compound"]
        log.info("  [PASS] VADER lexicon OK (sample compound=%.4f)", score)
    except LookupError:
        log.error(
            "  [FAIL] VADER lexicon not found.\n"
            "         Run: python setup_local.py --skip-model"
        )
        ok = False
    except Exception as exc:
        log.error("  [FAIL] VADER error: %s", exc)
        ok = False

    log.info(sep)
    if ok:
        log.info("  RESULT: ALL CHECKS PASSED — system is ready to run offline.")
    else:
        log.error("  RESULT: SOME CHECKS FAILED — re-run python setup_local.py")
    log.info(sep)
    return ok


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)

    log.info("=" * 60)
    log.info("   Sentiment Trader -- Local Setup")
    log.info("=" * 60)

    # --check: offline integrity only, no downloads
    if args.check:
        passed = check_integrity(model_dir)
        sys.exit(0 if passed else 1)

    errors: list[str] = []

    # ── DistilBERT ────────────────────────────────────────────────────────────
    if not args.skip_model:
        # Read model ID from .env / environment
        from dotenv import load_dotenv  # noqa: PLC0415
        import os  # noqa: PLC0415
        load_dotenv(ROOT / ".env", override=False)
        model_id = os.getenv(
            "HF_MODEL_NAME",
            "distilbert-base-uncased-finetuned-sst-2-english",
        )

        if model_dir.exists() and any(model_dir.iterdir()):
            log.info(
                "Model directory already exists and is non-empty (%s). "
                "Skipping download. Delete the directory to re-download.",
                model_dir,
            )
        else:
            try:
                download_model(model_id, model_dir)
            except Exception as exc:  # noqa: BLE001
                log.error("Model download failed: %s", exc)
                errors.append(f"Model download: {exc}")

        if model_dir.exists():
            try:
                verify_model(model_dir)
            except Exception as exc:  # noqa: BLE001
                log.warning("Model verification warning: %s", exc)
    else:
        log.info("Skipping DistilBERT download (--skip-model).")

    # ── VADER ─────────────────────────────────────────────────────────────────
    if not args.skip_vader:
        try:
            download_vader()
            verify_vader()
        except Exception as exc:  # noqa: BLE001
            log.error("VADER setup failed: %s", exc)
            errors.append(f"VADER: {exc}")
    else:
        log.info("Skipping VADER download (--skip-vader).")

    # -- Post-download integrity check ----------------------------------------
    log.info("")
    log.info("Running final integrity check...")
    check_integrity(model_dir)

    # -- Summary ---------------------------------------------------------------
    log.info("")
    log.info("=" * 60)
    if errors:
        log.info("  Setup completed WITH ERRORS:")
        log.info("=" * 60)
        for err in errors:
            log.error("  FAIL: %s", err)
        sys.exit(1)
    else:
        log.info("  Setup complete -- system is ready to run offline.")
        log.info("=" * 60)
        log.info("")
        log.info("Next step:")
        log.info("  streamlit run dashboard/app.py")



if __name__ == "__main__":
    main()
