cd ../..
# python train_ste.py --add_exp_version 1 --config configs/nerf_mic/mic_codec_ste_jpeg80.txt \
#                 --ckpt log/tensorf_mic_VM/tensorf_mic_VM.th\
#                 --compression --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project jpeg_ste \
#                 --n_iters 40000 --TV_weight_app 0.1 --refresh_k 32 \
#                 --save_every 10000 --vis_every 10000

# python train_ste.py --add_exp_version 1 --config configs/nerf_mic/mic_codec_ste_jpeg65.txt \
#                 --ckpt log/tensorf_mic_VM/tensorf_mic_VM.th\
#                 --compression --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project jpeg_ste \
#                 --n_iters 40000 --TV_weight_app 0.1 --refresh_k 32 \
#                 --save_every 10000 --vis_every 10000

# python train_ste.py --add_exp_version 1 --config configs/nerf_mic/mic_codec_ste_jpeg50.txt \
#                 --ckpt log/tensorf_mic_VM/tensorf_mic_VM.th\
#                 --compression --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project jpeg_ste \
#                 --n_iters 40000 --TV_weight_app 0.1 --refresh_k 32 \
#                 --save_every 10000 --vis_every 10000


python train_ste.py --add_exp_version 1 --config configs/nerf_mic/mic_codec_ste_jpeg35.txt \
                --ckpt log/tensorf_mic_VM/tensorf_mic_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 40000 --TV_weight_app 0.1 --refresh_k 32 \
                --save_every 10000 --vis_every 10000


