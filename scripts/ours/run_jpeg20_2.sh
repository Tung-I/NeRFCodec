cd ../..

python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg20.txt \
                --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 12000 \
                --save_every 4000 --vis_every 4000

python train_ste.py --add_exp_version 1 --config configs/nerf_materials/materials_codec_ste_jpeg20.txt \
                --ckpt log/tensorf_materials_VM/tensorf_materials_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 12000 \
                --save_every 4000 --vis_every 4000

python train_ste.py --add_exp_version 1 --config configs/nerf_mic/mic_codec_ste_jpeg20.txt \
                --ckpt log/tensorf_mic_VM/tensorf_mic_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 12000 \
                --save_every 4000 --vis_every 4000

python train_ste.py --add_exp_version 1 --config configs/nerf_ship/ship_codec_ste_jpeg20.txt \
                --ckpt log/tensorf_ship_VM/tensorf_ship_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 12000 \
                --save_every 4000 --vis_every 4000