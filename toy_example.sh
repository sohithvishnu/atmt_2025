#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=0:15:0
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=toy_example1.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# clean up from previous runs
rm -rf toy_example/data/prepared
rm -rf toy_example/tokenizers
rm -rf toy_example/checkpoints
rm -rf toy_example/logs
rm -f toy_example/toy_example_output.en

python preprocess.py \
    --source-lang cz \
    --target-lang en \
    --raw-data ./toy_example/data/raw \
    --dest-dir ./toy_example/data/prepared \
    --model-dir ./toy_example/tokenizers \
    --test-prefix test \
    --train-prefix train \
    --valid-prefix valid \
    --src-vocab-size 1000 \
    --tgt-vocab-size 1000 \
    --ignore-existing \
    --force-train
 
python train.py \
    --data toy_example/data/prepared/ \
    --src-tokenizer toy_example/tokenizers/cz-bpe-1000.model \
    --tgt-tokenizer toy_example/tokenizers/en-bpe-1000.model \
    --source-lang cz \
    --target-lang en \
    --batch-size 32 \
    --arch transformer \
    --max-epoch 10 \
    --log-file toy_example/logs/train.log \
    --save-dir toy_example/checkpoints/ \
    --ignore-checkpoints \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 100 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3 \
    --cuda 
python translate.py \
    --input toy_example/data/raw/test.cz \
    --src-tokenizer toy_example/tokenizers/cz-bpe-1000.model \
    --tgt-tokenizer toy_example/tokenizers/en-bpe-1000.model \
    --checkpoint-path toy_example/checkpoints/checkpoint_best.pt \
    --batch-size 1 \
    --max-len 100 \
    --output toy_example/toy_example_output.en \
    --bleu \
    --reference toy_example/data/raw/test.en \
    --cuda
