#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=05:20:00
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=translate_only_out.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/syacha/data/conda/envs/atmt/pkgs/cuda-toolkit

# TRANSLATE
python translate.py     --cuda     --input ~/shares/atomt.pilot.s3it.uzh/cz-en/data/raw/test.cz     --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model     --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model     --checkpoint-path cz-en/checkpoints/checkpoint_best.pt     --output output.txt     --max-len 300
