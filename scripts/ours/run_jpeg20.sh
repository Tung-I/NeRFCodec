cd ../..

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_jpeg20.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 12000 \
                --save_every 4000 --vis_every 4000

python train_ste.py --add_exp_version 1 --config configs/nerf_drums/drums_codec_ste_jpeg20.txt \
                --ckpt log/tensorf_drums_VM/tensorf_drums_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 12000 \
                --save_every 4000 --vis_every 4000

python train_ste.py --add_exp_version 1 --config configs/nerf_ficus/ficus_codec_ste_jpeg20.txt \
                --ckpt log/tensorf_ficus_VM/tensorf_ficus_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 12000 \
                --save_every 4000 --vis_every 4000

python train_ste.py --add_exp_version 1 --config configs/nerf_hotdog/hotdog_codec_ste_jpeg20.txt \
                --ckpt log/tensorf_hotdog_VM/tensorf_hotdog_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 12000 \
                --save_every 4000 --vis_every 4000






