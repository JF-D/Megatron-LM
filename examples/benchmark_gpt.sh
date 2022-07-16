#! /bin/bash

# model param
# GPT-1.5B
# nlayer=48
# seq_length=1024
# hidden_size=1600
# nheads=16
# model=gpt1.5b

# GPT-12L
# nlayer=12
# seq_length=1024
# hidden_size=1024
# nheads=16
# model=gpt12l

# GPT-2
nlayer=12
seq_length=1024
hidden_size=768
nheads=12
model=gpt2

# training param
# ndev=16
# global_bs=64
# micro_bs=2 # global_bs / dp_deg / n_macro_batch
# pp_deg=2
# mp_deg=1
# dp_deg=$(( $ndev / $(( $pp_deg * $mp_deg )) ))

ndev=$1
ps=$2
PROF=$3

g=$(($ndev<8?$ndev:8))

global_bs=$(( ${ndev} * 4 ))
n_micro_batch=1
pp_deg=1
if [ "${ps}" = "mp" ]; then
    if [ "${model}" = "gpt2" ]; then
        mp_deg=$(($ndev<4?$ndev:4))
    else
        mp_deg=$ndev
    fi
else
    mp_deg=1
fi
dp_deg=$(( $ndev / $(( $pp_deg * $mp_deg )) ))
micro_bs=$(( ${global_bs} / $(( ${dp_deg} * ${n_micro_batch} )) )) # global_bs / dp_deg / n_macro_batch

activation= #"--activations-checkpoint-method uniform"

vocab_size=40478

prefix=${model}_${ps}

CMD_PREFIX=""
CMD_SUFFIX=""
if [ "${PROF}" = "profile" ]; then
    export LD_LIBRARY_PATH=/nvme/platform/dep/cuda11.0-cudnn8.0/nsight-systems-2020.3.2/target-linux-x64:$LD_LIBRARY_PATH
    CMD_PREFIX="/nvme/platform/dep/cuda11.0-cudnn8.0/bin/nsys profile -t cudnn,cuda,osrt,nvtx \
                --output log/nvvp/gpt_n${ndev}_%q{SLURM_PROCID} --force-overwrite true"
    CMD_SUFFIX="--timeline"
fi

if [ "${ps}" = "dp" ]; then
    ddp_impl='torch'
else
    ddp_impl='local'
fi

#NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH \
# -w SH-IDC1-10-198-4-[81,82]
srun -p pat_test -x SH-IDC1-10-198-4-[185] -n $ndev --gres=gpu:$g --tasks-per-node=$g \
       ${CMD_PREFIX} python pretrain_gpt.py \
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
       --DDP-impl ${ddp_impl} $activation \
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
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --synthetic --launch slurm ${CMD_SUFFIX} \
       2>&1 | tee log/${prefix}_${ndev}.log
