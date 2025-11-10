#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_assignment2.out

# === Environment setup ===
set -e
module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# === Directory definitions ===
PROJECT_DIR=/home/syacha/data/atmt_2025/cz-en
RAW_DATA=/home/syacha/shares/cz-en/data/raw
MODEL_DIR=$PROJECT_DIR/tokenizers
DEST_DIR=$PROJECT_DIR/data/prepared
LOG_DIR=$PROJECT_DIR/logs
CKPT_DIR=$PROJECT_DIR/checkpoints
OUTPUT_FILE=$PROJECT_DIR/output.txt

mkdir -p $MODEL_DIR $DEST_DIR $LOG_DIR $CKPT_DIR

echo "🚀 Starting ATMT 2025 Joint-BPE training pipeline"
echo "-----------------------------------------------"

# ==============================================
# 🧩 STEP 1 — PREPARE DATA (Joint BPE)
# ==============================================
echo "📦 [1/3] Preprocessing Czech–English data..."
python preprocess.py \
  --source-lang cz \
  --target-lang en \
  --raw-data $RAW_DATA \
  --dest-dir $DEST_DIR \
  --model-dir $MODEL_DIR \
  --train-prefix train \
  --valid-prefix valid \
  --test-prefix test \
  --src-vocab-size 8000 \
  --tgt-vocab-size 8000 \
  --src-model $MODEL_DIR/joint-bpe-8000.model \
  --tgt-model $MODEL_DIR/joint-bpe-8000.model \
  --joint-bpe \
  --force-train

echo "✅ Data preprocessing complete."
echo

# ==============================================
# 🧠 STEP 2 — TRAIN MODEL
# ==============================================
echo "🎓 [2/3] Training transformer model..."
python train.py \
  --cuda \
  --data $DEST_DIR \
  --src-tokenizer $MODEL_DIR/joint-bpe-8000.model \
  --tgt-tokenizer $MODEL_DIR/joint-bpe-8000.model \
  --source-lang cz \
  --target-lang en \
  --batch-size 32 \
  --arch transformer \
  --max-epoch 1 \
  --log-file $LOG_DIR/train.log \
  --save-dir $CKPT_DIR \
  --ignore-checkpoints \
  --encoder-dropout 0.1 \
  --decoder-dropout 0.1 \
  --dim-embedding 256 \
  --attention-heads 4 \
  --dim-feedforward-encoder 1024 \
  --dim-feedforward-decoder 1024 \
  --max-seq-len 256 \
  --n-encoder-layers 3 \
  --n-decoder-layers 3

echo "✅ Model training complete. Checkpoints saved to $CKPT_DIR."
echo

# ==============================================
# 🗣️ STEP 3 — TRANSLATE TEST SET
# ==============================================
echo "🗣️ [3/3] Translating test set..."
python translate.py \
  --cuda \
  --input $DEST_DIR/test.cz \
  --src-tokenizer $MODEL_DIR/joint-bpe-8000.model \
  --tgt-tokenizer $MODEL_DIR/joint-bpe-8000.model \
  --checkpoint-path $CKPT_DIR/checkpoint_best.pt \
  --output $OUTPUT_FILE \
  --max-len 300

echo "✅ Translation completed successfully."
echo "📄 Output written to: $OUTPUT_FILE"
echo "-----------------------------------------------"
echo "🏁 Pipeline finished."
