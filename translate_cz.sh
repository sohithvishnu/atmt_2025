#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=translate_cz.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

python translate.py \
    --cuda \
    --input task3_data/cs_combined.txt \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output task3_data/output.txt \
    --batch-size 32 \
    --max-len 128 \
    --bleu \
    --reference task3_data/en_combined.txt
