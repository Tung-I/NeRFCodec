cd ../..


python train_ste.py --add_exp_version 1 --config configs/tt_family/family_codec_ste_jpeg10.txt \
                --ckpt log/tensorf_family_VM/tensorf_family_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 32 \
                --save_every 10000 --vis_every 10000

python train_ste.py --add_exp_version 1 --config configs/tt_family/family_codec_ste_jpeg20.txt \
                --ckpt log/tensorf_family_VM/tensorf_family_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 32 \
                --save_every 10000 --vis_every 10000

