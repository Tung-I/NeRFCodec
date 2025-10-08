cd ../..
python noise_train.py --add_exp_version 1 \
                --expname noise_quant_light \
                --wandb_project LegoNoise \
                --config configs/lego_noise.txt \
                --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th\
                --render_test 1 --batch_size 65536 \
                --lr_decay_target_ratio 1 \
                --compression \
                --codec_noise \
                --codec_noise_seed 0 \
                --codec_noise_mode quant \
                --codec_noise_level 1

python noise_train.py --add_exp_version 1 \
                --expname noise_quant_medium \
                --wandb_project LegoNoise \
                --config configs/lego_noise.txt \
                --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th\
                --render_test 1 --batch_size 65536 \
                --lr_decay_target_ratio 1 \
                --compression \
                --codec_noise \
                --codec_noise_seed 0 \
                --codec_noise_mode quant \
                --codec_noise_level 2

python noise_train.py --add_exp_version 1 \
                --expname noise_quant_heavy \
                --wandb_project LegoNoise \
                --config configs/lego_noise.txt \
                --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th\
                --render_test 1 --batch_size 65536 \
                --lr_decay_target_ratio 1 \
                --compression \
                --codec_noise \
                --codec_noise_seed 0 \
                --codec_noise_mode quant \
                --codec_noise_level 3