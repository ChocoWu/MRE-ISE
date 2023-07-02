#!/usr/bin/env bash

DATASET_NAME="MRE"
PRETRAIN_NAME="openai/clip-vit-base-patch32"

CUDA_VISIBLE_DEVICES=2 python -u run.py \
        --pretrain_name=${PRETRAIN_NAME} \
        --dataset_name=${DATASET_NAME} \
        --num_epochs=30 \
        --batch_size=16 \
        --lr_pretrained=2e-5 \
        --lr_main=2e-4 \
        --warmup_ratio=0.01 \
        --eval_begin_epoch=10 \
        --seed=1234 \
        --do_train \
        --max_seq=40 \
        --max_obj=40 \
        --beta=0.01 \
        --temperature=0.1 \
        --eta1=0.8 \
        --eta2=0.7 \
        --neighbor_num=2 \
        --topic_keywords_number=10 \
        --topic_number=10 \
        --save_path="ckpt"