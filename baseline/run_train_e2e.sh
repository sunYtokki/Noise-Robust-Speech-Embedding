#!/bin/bash
ssl_type=wavlm-large

# Train
pool_type=AttentiveStatisticsPooling
model_path=/proj/speech/users/syk2145/baseline/MSP-Podcast_Challenge/baseline/model/byol_wavlm_large-snr2_20-ema_997_noisy_e2e_ft2
config_path=config_cat_noisy.json
for seed in 7; do
    python train_ft_cat_ser_weighted_checkpoint.py \
        --seed=${seed} \
        --ssl_type=${ssl_type} \
        --batch_size=32 \
        --accumulation_steps=4 \
        --lr=5e-6 \
        --epochs=10 \
        --pooling_type=${pool_type} \
        --config=${config_path} \
        --freeze_ssl=False \
        --model_path=${model_path} || exit 0;

done
