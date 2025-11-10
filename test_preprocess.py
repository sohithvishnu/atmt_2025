import os
import tempfile
import pickle
import shutil
from subprocess import run


def create_dummy_data():
    """Creates temporary dummy bilingual data for testing."""
    tmpdir = tempfile.mkdtemp()
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Minimal bilingual dataset
    src_lines = ["I like apples .", "She loves bananas .", "We eat fruits ."]
    tgt_lines = ["Ich mag Äpfel .", "Sie liebt Bananen .", "Wir essen Früchte ."]

    with open(os.path.join(data_dir, "train.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(src_lines))
    with open(os.path.join(data_dir, "train.de"), "w", encoding="utf-8") as f:
        f.write("\n".join(tgt_lines))

    return tmpdir, data_dir


def run_preprocess(args):
    """Helper to run preprocess.py as subprocess."""
    result = run(args, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Preprocessing failed:\n{result.stderr}")


def read_vocab(vocab_path):
    """Reads SentencePiece vocab file and returns set of subword tokens."""
    tokens = set()
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip().split("\t")[0]
            if token:
                tokens.add(token)
    return tokens


def compute_vocab_overlap(vocab_src, vocab_tgt):
    """Compute and print shared subword overlap."""
    shared = vocab_src & vocab_tgt
    overlap_ratio = len(shared) / max(len(vocab_src | vocab_tgt), 1)
    print(f"🔍 Shared subwords: {len(shared)} / {len(vocab_src | vocab_tgt)} "
          f"({overlap_ratio * 100:.2f}%)")
    if len(shared) > 0:
        print(f"   Example shared tokens: {list(shared)[:10]}")
    return overlap_ratio


def test_joint_bpe_pipeline():
    """Test preprocessing with the joint BPE flag."""
    tmpdir, data_dir = create_dummy_data()
    model_dir = os.path.join(tmpdir, "models")
    dest_dir = os.path.join(tmpdir, "bin")
    os.makedirs(model_dir, exist_ok=True)

    vocab_size = 100
    cmd = [
        "python", "preprocess.py",
        "--raw-data", data_dir,
        "--train-prefix", "train",
        "--source-lang", "en",
        "--target-lang", "de",
        "--dest-dir", dest_dir,
        "--model-dir", model_dir,
        "--joint-bpe",
        "--src-vocab-size", str(vocab_size),
        "--force-train"
    ]

    run_preprocess(cmd)

    # Check model and vocab exist
    model_file = os.path.join(model_dir, f"joint-bpe-{vocab_size}.model")
    vocab_file = os.path.join(model_dir, f"joint-bpe-{vocab_size}.vocab")
    assert os.path.exists(model_file), f"Joint model file not created: {model_file}"
    assert os.path.exists(vocab_file), f"Joint vocab file not created: {vocab_file}"

    # Check binary outputs
    train_en = os.path.join(dest_dir, "train.en")
    train_de = os.path.join(dest_dir, "train.de")
    assert os.path.exists(train_en), "Encoded train.en missing!"
    assert os.path.exists(train_de), "Encoded train.de missing!"

    # Ensure pickle data loads correctly
    for fpath in [train_en, train_de]:
        with open(fpath, "rb") as f:
            data = pickle.load(f)
            assert isinstance(data, list)
            assert all(isinstance(x, list) or hasattr(x, "__len__") for x in data)

    # Analyze vocab overlap (should be 100% for joint BPE)
    vocab = read_vocab(vocab_file)
    print("\n📊 Joint BPE Vocabulary Stats:")
    print(f"   Total tokens: {len(vocab)}")
    print(f"   Sample tokens: {list(vocab)[:10]}")

    print("✅ Joint BPE preprocessing test passed successfully.")
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_separate_bpe_pipeline():
    """Test preprocessing with separate source/target tokenizers."""
    tmpdir, data_dir = create_dummy_data()
    model_dir = os.path.join(tmpdir, "models_sep")
    dest_dir = os.path.join(tmpdir, "bin_sep")
    os.makedirs(model_dir, exist_ok=True)

    vocab_size = 200  # smaller vocab avoids "too high" error for tiny corpus
    cmd = [
        "python", "preprocess.py",
        "--raw-data", data_dir,
        "--train-prefix", "train",
        "--source-lang", "en",
        "--target-lang", "de",
        "--dest-dir", dest_dir,
        "--model-dir", model_dir,
        "--src-vocab-size", str(vocab_size),
        "--tgt-vocab-size", str(vocab_size),
        "--force-train"
    ]

    run_preprocess(cmd)

    # Check individual models exist
    model_en = os.path.join(model_dir, f"en-bpe-{vocab_size}.model")
    model_de = os.path.join(model_dir, f"de-bpe-{vocab_size}.model")
    vocab_en = os.path.join(model_dir, f"en-bpe-{vocab_size}.vocab")
    vocab_de = os.path.join(model_dir, f"de-bpe-{vocab_size}.vocab")

    assert os.path.exists(model_en), f"Source model not created: {model_en}"
    assert os.path.exists(model_de), f"Target model not created: {model_de}"

    # Compute shared subword overlap (expect < joint BPE)
    vocab_src = read_vocab(vocab_en)
    vocab_tgt = read_vocab(vocab_de)
    print("\n📊 Separate BPE Vocabulary Stats:")
    compute_vocab_overlap(vocab_src, vocab_tgt)

    print("✅ Separate BPE preprocessing test passed successfully.")
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("🚀 Starting preprocessing tests...\n")
    try:
        test_joint_bpe_pipeline()
        test_separate_bpe_pipeline()
        print("\n🎉 All preprocessing tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
