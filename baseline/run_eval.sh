#!/bin/bash
ssl_type=wavlm-large

# Train
pool_type=AttentiveStatisticsPooling
model_path=//proj/speech/users/syk2145/nrse/checkpoints/baseline/byol_wavlm_large-snr2_20-ema_997_noisy_e2e_ft_gradual
for seed in 7; do
    
    python eval_cat_ser_weighted_with_list.py \
        --ssl_type=${ssl_type} \
        --pooling_type=${pool_type} \
        --model_path=${model_path}  \
        --audio_list=/proj/speech/users/syk2145/nrse/labels/processed/msp1_11-test2-snr8_processed.txt \
        --audio_dir=/proj/speech/users/syk2145/data \
        --store_path=result/weight_cat_ser/wavLM_adamW/${seed}.txt || exit 0;

done
