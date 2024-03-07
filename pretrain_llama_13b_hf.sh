#!/bin/bash
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export WANDB_API_KEY=5da89edeb8bd06c5cfdf131b4d7eb33ff7e55782
export CUDA_HOME="/ssd/cip/miniconda3/envs/dm-torch2.1.2-cu11.8/"
# export NCCL_DEBUG=INFO
# continue to pretrain bloom1b1 on a single node
# Multi-node will require either a `hostfile` or switching to `torch.distributed.launch`


function rand() {
  min=$1
  max=$(($2 - $min + 1))
  num=$(($RANDOM + 1000000000)) #增加一个10位的数再求余
  echo $(($num % $max + $min))
}

function get_gpu_num() {
  IFS=,
  num=0
  for i in ${NVIDIA_VISIBLE_DEVICES}; do
    num=$((${num} + 1))
  done
  echo ${num}
  return ${num}
}


proj_ent=cipsup
proj_name=dm-sft
proj_group=llama-2-13b-16k
run_name=tp8_pp2_zero1_gbs128-v0_16k

mbs=1
gbs=128
N_GPUS=$(get_gpu_num)
PORT=$(rand 10000 50000)

DATA_PATH=5.84 /data/dataset/zhuque-data/tokenized/ArXiv/zhuque-ArXiv-aa_content_document 2.57 /data/dataset/zhuque-data/tokenized/Baike/zhuque-Baike-aa_content_document 4.46 /data/dataset/zhuque-data/tokenized/Book_en/zhuque-Book_en-aa_content_document 2.18 /data/dataset/zhuque-data/tokenized/Book_zh/zhuque-Book_zh-aa_content_document 7.13 /data/dataset/zhuque-data/tokenized/Code/zhuque-Code-aa_content_document 0.72 /data/dataset/zhuque-data/tokenized/Code/zhuque-Code-ab_content_document 4.55 /data/dataset/zhuque-data/tokenized/CommonCrawl_550G/zhuque-CommonCrawl_550G-aa_content_document 4.65 /data/dataset/zhuque-data/tokenized/CommonCrawl_550G/zhuque-CommonCrawl_550G-ab_content_document 4.55 /data/dataset/zhuque-data/tokenized/CommonCrawl_550G/zhuque-CommonCrawl_550G-ac_content_document 4.55 /data/dataset/zhuque-data/tokenized/CommonCrawl_550G/zhuque-CommonCrawl_550G-ad_content_document 4.55 /data/dataset/zhuque-data/tokenized/CommonCrawl_550G/zhuque-CommonCrawl_550G-ae_content_document 4.26 /data/dataset/zhuque-data/tokenized/CommonCrawl_550G/zhuque-CommonCrawl_550G-af_content_document 0.15 /data/dataset/zhuque-data/tokenized/Exam/zhuque-Exam-aa_content_document 0.04 /data/dataset/zhuque-data/tokenized/Gov/zhuque-Gov-aa_content_document 4.85 /data/dataset/zhuque-data/tokenized/Law/zhuque-Law-aa_content_document 3.27 /data/dataset/zhuque-data/tokenized/Law/zhuque-Law-ab_content_document 4.26 /data/dataset/zhuque-data/tokenized/News/zhuque-News-aa_content_document 4.16 /data/dataset/zhuque-data/tokenized/News/zhuque-News-ab_content_document 4.26 /data/dataset/zhuque-data/tokenized/News/zhuque-News-ac_content_document 1.98 /data/dataset/zhuque-data/tokenized/News/zhuque-News-ad_content_document 4.36 /data/dataset/zhuque-data/tokenized/OpenWebText_300G/zhuque-OpenWebText_300G-aa_content_document 4.36 /data/dataset/zhuque-data/tokenized/OpenWebText_300G/zhuque-OpenWebText_300G-ab_content_document 4.36 /data/dataset/zhuque-data/tokenized/OpenWebText_300G/zhuque-OpenWebText_300G-ac_content_document 1.09 /data/dataset/zhuque-data/tokenized/OpenWebText_300G/zhuque-OpenWebText_300G-ad_content_document 0.89 /data/dataset/zhuque-data/tokenized/Patent/zhuque-Patent-aa_content_document 4.85 /data/dataset/zhuque-data/tokenized/StackExchange/zhuque-StackExchange-aa_content_document 6.24 /data/dataset/zhuque-data/tokenized/Wikipedia/zhuque-Wikipedia-aa_content_document 0.88 /data/dataset/zhuque-data/tokenized/Wikipedia/zhuque-Wikipedia-ab_content_document 0.11 /data/dataset/zhuque-data/tokenized/Wikipedia_zh/zhuque-Wikipedia_zh-aa_content_document # data of pretraining
LOAD_PATH=/data/models/Zhuque2-13B-Base     # the path where checkpoint will be saved and load from
SAVE_PATH=/data/code/xinchunlei/checkpoint/llama-2-13b-16k            # the path where checkpoint will be saved
DS_PATH=/data/code/xinchunlei/megatron-deepspeed/scripts/ds-cofig.json


DISTRIBUTE_ARGS=" \
    --deepspeed \
    --deepspeed_config $DS_PATH
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 2 \
    --distributed-backend nccl \
    --pp-partition-method type:transformer|embedding \
    "

OPTIMIZE_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --weight-decay 1e-1 \
    --lr 2e-5 \
    --min-lr 2e-6 \
    --lr-decay-style cosine \
    --lr-warmup-fraction .01 \
    --lr-decay-iters 30000 \
    --clip-grad 1.0 \
    --micro-batch-size 1 \
    --train-iters 3 \
    --data-impl mmap \
    --split 949,50,1 \
    "

TRAINING_ARG=" \
    --bf16 \
    --rope-scaling \
    --rope-scaling-type dynamic \
    --rope-scaling-factor 4. \
    --use-efficient-attention \
    --attention-softmax-in-fp32 \
  "

OPTIMIZE_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --weight-decay 0. \--clip-grad 1.0 \
    "


GPT_ARGS=" \
    --num-layers 40 \
    --hidden-size 5120 \
    --num-attention-heads 40 \
    --init-method-std 0.0088 \
    --seq-length 16384 \
    --max-position-embeddings 4096 \
    --pad-vocab-size-to 64000 \
    --tokenizer-name-or-path /data/hf_models/Zhuque2-13B-Base \
    --make-vocab-size-divisible-by 1 \
    --sync-tp-duplicated-parameters \
    --seed 24 \
     "
# wandb_args=" \
#     --to-wandb \
#     --wandb-log-interval 1 \
#     --wandb-project ${proj_name} \
#     --wandb-entity ${proj_ent} \
#     --wandb-group ${proj_group} \
#     --wandb-run-name ${run_name} \
# "


OUTPUT_ARGS=" \
    --save-interval 1000 \
    --log-interval 1 \
    --tensorboard-log-interval 100 \
    --tensorboard-dir $SAVE_PATH/tensorboard \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    $wandb_args \
    "

DATA_ARGS=" \
    --epoch 3 \
    --split 100,0,0 \
    --shuffle-all-epoch \
    --data-impl mmap \
    --data-path $DATA_PATH \
    --micro-batch-size ${mbs} \
    --global-batch-size ${gbs} \
    --load $LOAD_PATH \
    --save $SAVE_PATH \
    --load-checkpoint-type hf \
    "


ALL_ARGS="$DISTRIBUTE_ARGS $TRAINING_ARG $GPT_ARGS $OPTIMIZE_ARGS $OUTPUT_ARGS $DATA_ARGS"

LAUNCHER="deepspeed --num_gpus $N_GPUS --master_port $PORT"

CMD="$LAUNCHER pretrain_llama.py $ALL_ARGS"

echo $CMD

$CMD



