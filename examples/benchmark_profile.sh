#! /bin/bash

# model param
nlayer=6
seq_length=1024
hidden_size=1024
nheads=16

# training param
ndev=2
global_bs=4
micro_bs=4
pp_deg=2
mp_deg=1
dp_deg=$(( $ndev / $(( $pp_deg \* $mp_deg )) ))

activation= #"--activations-checkpoint-method uniform"

vocab_size=40478

sudo mpirun -np $ndev --allow-run-as-root \
       $(which nvprof) --profile-from-start off \
       -o log/gpt_n8_%p.nvvp -f \
       /home/duanjiangfei/env/miniconda3/envs/pt1.8v1/bin/python pretrain_gpt.py \
       --num-layers $nlayer \
       --seq-length $seq_length \
       --max-position-embeddings $seq_length \
       --hidden-size $hidden_size \
       --num-attention-heads $nheads \
       --vocab-size $vocab_size \
       --micro-batch-size $micro_bs --global-batch-size $global_bs \
       --no-masked-softmax-fusion --no-bias-gelu-fusion --no-bias-dropout-fusion \
       --openai-gelu --no-optimizer-fusion --no-layernorm-fusion \
       --no-async-tensor-model-parallel-allreduce \
       --DDP-impl local $activation \
       --pipeline-model-parallel-size $pp_deg \
       --tensor-model-parallel-size $mp_deg \
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
       --no-async-tensor-model-parallel-allreduce \
       --synthetic --no-compile --timeline
