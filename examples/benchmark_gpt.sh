#! /bin/bash

# Runs the "345M" parameter model

nlayer=12
seq_length=1024
hidden_size=1024
nheads=16
bs=4

vocab_size=40478


mpirun -np 8 python pretrain_gpt.py \
       --num-layers $nlayer \
       --seq-length $seq_length \
       --max-position-embeddings $seq_length \
       --hidden-size $hidden_size \
       --num-attention-heads $nheads \
       --vocab-size $vocab_size \
       --micro-batch-size $bs \
       --no-masked-softmax-fusion --no-bias-gelu-fusion --no-bias-dropout-fusion \
       --openai-gelu --no-optimizer-fusion --no-layernorm-fusion \
       --DDP-impl torch \
       --pipeline-model-parallel-size 1 \
       --tensor-model-parallel-size 4 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 10 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --launch mpirun \
       --synthetic --timeline


# mpirun -np 1 python pretrain_gpt.py \
#        --num-layers 24 \
#        --seq-length 1024 \
#        --max-position-embeddings 1024 \
#        --hidden-size 1024 \
#        --num-attention-heads 16 \
#        --micro-batch-size 4 \
#        --global-batch-size 4 \
#        --no-masked-softmax-fusion --no-bias-gelu-fusion --no-bias-dropout-fusion \
#        --openai-gelu --no-optimizer-fusion \
#        --pipeline-model-parallel-size 1 \
#        --tensor-model-parallel-size 1 \
#        --vocab-size 50257 \
#        --train-iters 500000 \
#        --lr-decay-iters 320000 \
#        --vocab-file gpt2-vocab.json \
#        --merge-file gpt2-merges.txt \
#        --data-impl mmap \
#        --split 949,50,1 \
#        --distributed-backend nccl \
#        --lr 0.00015 \
#        --min-lr 1.0e-5 \
#        --lr-decay-style cosine \
#        --weight-decay 1e-2 \
#        --clip-grad 1.0 \
#        --lr-warmup-fraction .01 \
#        --activations-checkpoint-method uniform \
#        --log-interval 10 \
#        --save-interval 10000 \
#        --eval-interval 1000 \
#        --eval-iters 10 \
#        --launch mpirun \
#        --synthetic
