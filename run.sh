#!/usr/bin/bash



##=== MeanTeacherv2 ===
# cifar10-4k
CUDA_VISIBLE_DEVICES=$1 python3 main.py --num-labels=5000  --usp-weight=20.0 --ema-decay=0.99 --optim=adam --epochs=200 --lr=0.00005 --momentum=0.9 --weight-decay=5e-4 --nesterov=True --lr-scheduler=cos-warmup --min-lr=1e-7 --rampup-length=15 --rampdown-length=50 --save-freq=100 2>&1 | tee results/mtv2_cifar10-4k_$(date +%y-%m-%d-%H-%M).txt

