import os
import logging
import sentencepiece as spm


class BPETokenizer:
    def __init__(self,
                 language,
                 vocab_size=32000,
                 eos="</s>",
                 bos="<s>",
                 pad="<pad>",
                 unk="<unk>"):
        self.language = language
        self.vocab_size = vocab_size
        self.eos = eos
        self.bos = bos
        self.pad = pad
        self.unk = unk
        self.tokenizer = None

    # -----------------------------
    # 🧩 Train the SentencePiece tokenizer
    # -----------------------------
    def train_tokenizer(self, training_data, model_dir):
        model_name = os.path.join(model_dir, f"{self.language}-bpe-{self.vocab_size}")
        cmd = (
            f"--input={training_data} "
            f"--model_prefix={model_name} "
            f"--pad_id=3 "
            f"--vocab_size={self.vocab_size} "
            f"--model_type=bpe "
            f"--unk_piece={self.unk} "
            f"--bos_piece={self.bos} "
            f"--eos_piece={self.eos} "
            f"--pad_piece={self.pad}"
        )

        logging.info(f"🚀 Training SentencePiece model for {self.language} "
                     f"with target vocab size {self.vocab_size}...")

        try:
            spm.SentencePieceTrainer.train(cmd)
            logging.info(f"✅ Trained SentencePiece model for {self.language}")
        except RuntimeError as e:
            # Handle small corpus case
            if "Vocabulary size too high" in str(e):
                logging.warning(
                    f"⚠️ Requested vocab size {self.vocab_size} too high for small corpus. "
                    f"Retrying with reduced vocab size..."
                )
                smaller_vocab = max(50, self.vocab_size // 2)
                retry_cmd = cmd.replace(
                    f"--vocab_size={self.vocab_size}", f"--vocab_size={smaller_vocab}"
                )
                spm.SentencePieceTrainer.train(retry_cmd)
                self.vocab_size = smaller_vocab
                logging.info(f"✅ Retrained tokenizer for {self.language} "
                             f"with smaller vocab size {self.vocab_size}")
            else:
                raise

    # -----------------------------
    # 🧩 Load an existing tokenizer
    # -----------------------------
    def load(self, model_path):
        self.tokenizer = spm.SentencePieceProcessor(model_file=model_path)
        logging.info(f"✅ Loaded SentencePiece model from {model_path}")

    # -----------------------------
    # 🧩 Save vocab
    # -----------------------------
    def save_vocab(self, model_dir):
        model_name = os.path.join(model_dir, f"{self.language}-bpe-{self.vocab_size}.vocab")
        if self.tokenizer is None:
            model_path = os.path.join(model_dir, f"{self.language}-bpe-{self.vocab_size}.model")
            if not os.path.exists(model_path):
                logging.warning(f"No tokenizer found for {self.language}. Skipping vocab save.")
                return
            self.tokenizer = spm.SentencePieceProcessor(model_file=model_path)

        with open(model_name, "w", encoding="utf-8") as vocab_file:
            for i in range(self.tokenizer.get_piece_size()):
                piece = self.tokenizer.id_to_piece(i)
                score = self.tokenizer.get_score(i)
                vocab_file.write(f"{piece}\t{score}\n")

        logging.info(f"💾 Saved vocab for {self.language} -> {model_name}")

    # -----------------------------
    # 🧩 Encode text into tensor-like list
    # -----------------------------
    def encode_to_tensor(self, text, append_eos=True, consumer=None):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        ids = self.tokenizer.encode(text)
        if append_eos:
            ids.append(self.tokenizer.eos_id())
        if consumer:
            for idx in ids:
                consumer(idx)
        return ids


# ----------------------------------------------------
# 🧪 Quick test when running this file directly
# ----------------------------------------------------
if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    tmpdir = tempfile.mkdtemp()
    model_dir = os.path.join(tmpdir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Create a small dummy text file
    dummy_text = os.path.join(tmpdir, "train.en")
    with open(dummy_text, "w", encoding="utf-8") as f:
        f.write("I like apples .\nShe loves bananas .\nWe eat fruits .")

    # Initialize and train tokenizer
    tok = BPETokenizer(language="en", vocab_size=200)
    tok.train_tokenizer(training_data=dummy_text, model_dir=model_dir)

    # Load model and encode text
    model_path = os.path.join(model_dir, f"en-bpe-{tok.vocab_size}.model")
    tok.load(model_path)

    sample = "I love bananas ."
    encoded = tok.encode_to_tensor(sample)
    print(f"\n🧠 Encoded sample: {sample}")
    print(f"➡️ Tokens: {encoded}")

    # Save vocab file
    tok.save_vocab(model_dir)
    print("\n✅ Vocab saved successfully.")

    print(f"\nTemporary directory: {tmpdir}")
