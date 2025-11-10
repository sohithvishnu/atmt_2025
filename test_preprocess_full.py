#!/usr/bin/env python3
import os
import sys
import subprocess
import pickle
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

# === CONFIG (match your SLURM file) ===
SRC_LANG = "cz"
TGT_LANG = "en"

RAW_DATA = os.path.expanduser("~/shares/cz-en/data/raw")
DEST_DIR = "./cz-en/data/prepared"
MODEL_DIR = "./cz-en/tokenizers"
LOG_DIR = "./cz-en/logs"
CHECKPOINT_DIR = "./cz-en/checkpoints"
OUTPUT_FILE = "./cz-en/output.txt"

SRC_MODEL = os.path.join(MODEL_DIR, f"{SRC_LANG}-bpe-8000.model")
TGT_MODEL = os.path.join(MODEL_DIR, f"{TGT_LANG}-bpe-8000.model")

# === HELPER ===
def run_command(cmd, desc):
    """Run a subprocess and raise if it fails."""
    logging.info(f"\n🚀 Running: {desc}")
    logging.info(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    logging.info(result.stdout)
    if result.returncode != 0:
        logging.error(result.stderr)
        raise RuntimeError(f"❌ {desc} failed.")
    logging.info(f"✅ {desc} completed successfully.\n")
    return result


# === STEP 1: PREPROCESS ===
def test_preprocessing():
    cmd = [
        "python", "preprocess.py",
        "--source-lang", SRC_LANG,
        "--target-lang", TGT_LANG,
        "--raw-data", RAW_DATA,
        "--dest-dir", DEST_DIR,
        "--model-dir", MODEL_DIR,
        "--test-prefix", "test",
        "--train-prefix", "train",
        "--valid-prefix", "valid",
        "--src-vocab-size", "8000",
        "--tgt-vocab-size", "8000",
        "--force-train"
    ]
    run_command(cmd, "Data preprocessing")

    # --- Assertions ---
    assert os.path.exists(SRC_MODEL), f"Missing source model: {SRC_MODEL}"
    assert os.path.exists(TGT_MODEL), f"Missing target model: {TGT_MODEL}"

    # check dataset files
    for prefix in ["train", "valid", "test"]:
        for lang in [SRC_LANG, TGT_LANG]:
            fpath = os.path.join(DEST_DIR, f"{prefix}.{lang}")
            assert os.path.exists(fpath), f"Missing preprocessed file: {fpath}"
            with open(fpath, "rb") as f:
                data = pickle.load(f)
                assert isinstance(data, list) and len(data) > 0
    logging.info("📦 Preprocessing tests passed (vocab + data files verified).")


# === STEP 2: TRAIN ===
def test_training():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    cmd = [
        "python", "train.py",
        "--cuda",
        "--data", DEST_DIR,
        "--src-tokenizer", SRC_MODEL,
        "--tgt-tokenizer", TGT_MODEL,
        "--source-lang", SRC_LANG,
        "--target-lang", TGT_LANG,
        "--batch-size", "64",
        "--arch", "transformer",
        "--max-epoch", "1",  # use 1 epoch for testing
        "--log-file", os.path.join(LOG_DIR, "train_test.log"),
        "--save-dir", CHECKPOINT_DIR,
        "--ignore-checkpoints",
        "--encoder-dropout", "0.1",
        "--decoder-dropout", "0.1",
        "--dim-embedding", "256",
        "--attention-heads", "4",
        "--dim-feedforward-encoder", "1024",
        "--dim-feedforward-decoder", "1024",
        "--max-seq-len", "300",
        "--n-encoder-layers", "3",
        "--n-decoder-layers", "3"
    ]

    run_command(cmd, "Model training (1 epoch test)")

    # check checkpoint
    ckpt_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
    assert len(ckpt_files) > 0, "No checkpoint file saved!"
    logging.info(f"🧠 Training test passed ({len(ckpt_files)} checkpoints saved).")


# === STEP 3: TRANSLATION ===
def test_translation():
    ckpt_best = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pt")
    ckpt_any = None

    # pick the first checkpoint if _best doesn't exist
    if not os.path.exists(ckpt_best):
        ckpt_list = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
        if ckpt_list:
            ckpt_any = os.path.join(CHECKPOINT_DIR, ckpt_list[0])
        else:
            raise FileNotFoundError("No checkpoint found for translation.")
    ckpt_path = ckpt_best if os.path.exists(ckpt_best) else ckpt_any

    cmd = [
        "python", "translate.py",
        "--cuda",
        "--input", DEST_DIR,
        "--src-tokenizer", SRC_MODEL,
        "--tgt-tokenizer", TGT_MODEL,
        "--checkpoint-path", ckpt_path,
        "--output", OUTPUT_FILE,
        "--max-len", "300"
    ]

    run_command(cmd, "Model translation")

    # assert translation output exists and non-empty
    assert os.path.exists(OUTPUT_FILE), "Translation output missing!"
    with open(OUTPUT_FILE, "r") as f:
        lines = f.readlines()
        assert len(lines) > 0, "Translation output is empty!"
    logging.info("🗣️ Translation test passed.")


# === MAIN ===
if __name__ == "__main__":
    start_time = time.time()
    logging.info("🚀 Starting full ATMT pipeline test...\n")

    try:
        test_preprocessing()
        test_training()
        test_translation()
        logging.info(f"\n🎉 Full pipeline test completed successfully in {time.time() - start_time:.2f}s.")
    except Exception as e:
        logging.error(f"\n❌ Test failed: {e}")
        sys.exit(1)
