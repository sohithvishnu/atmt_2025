#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_assignment2_resume.out

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
OUTPUT_FILE=$PROJECT_DIR/output_avg.txt

mkdir -p $LOG_DIR $CKPT_DIR

echo "🚀 Resuming Czech–English Joint-BPE training"
echo "--------------------------------------------"
# ==============================================
# 🧠 STEP 2 — TRAIN MODEL (with checkpointing)
# ==============================================
echo "🎓 [2/3] Training transformer model with checkpointing..."
python train.py \
  --cuda \
  --data $DEST_DIR \
  --src-tokenizer $MODEL_DIR/joint-bpe-8000.model \
  --tgt-tokenizer $MODEL_DIR/joint-bpe-8000.model \
  --source-lang cz \
  --target-lang en \
  --batch-size 32 \
  --arch transformer \
  --max-epoch 2 \
  --log-file $LOG_DIR/train.log \
  --save-dir $CKPT_DIR \
  --save-interval 1 \
  --epoch-checkpoints \
  --encoder-dropout 0.1 \
  --decoder-dropout 0.1 \
  --dim-embedding 256 \
  --attention-heads 4 \
  --dim-feedforward-encoder 1024 \
  --dim-feedforward-decoder 1024 \
  --max-seq-len 256 \
  --n-encoder-layers 3 \
  --n-decoder-layers 3 \
  --restore-file $CKPT_DIR/checkpoint_last.pt 
echo "✅ Model training complete. Checkpoints saved to $CKPT_DIR."
echo

# ==============================================
# 💾 STEP 2.5 — AVERAGE LAST 3 CHECKPOINTS
# ==============================================
echo "🧮 Averaging last 3 checkpoints..."
python average_checkpoints.py \
  --checkpoint-dir $CKPT_DIR \
  --num-last 3 \
  --output $CKPT_DIR/checkpoint_avg.pt

echo "✅ Averaged checkpoint saved as checkpoint_avg.pt."
echo

# ==============================================
# 🗣️ STEP 3 — TRANSLATE TEST SET
# ==============================================
echo "🗣️ [3/3] Translating test set using averaged checkpoint..."
python translate.py \
  --cuda \
  --input $DEST_DIR/test.cz \
  --src-tokenizer $MODEL_DIR/joint-bpe-8000.model \
  --tgt-tokenizer $MODEL_DIR/joint-bpe-8000.model \
  --checkpoint-path $CKPT_DIR/checkpoint_avg.pt \
  --output $OUTPUT_FILE \
  --max-len 300

echo "✅ Translation completed successfully."
echo "📄 Output written to: $OUTPUT_FILE"
echo "-----------------------------------------------"
echo "🏁 Pipeline finished."
