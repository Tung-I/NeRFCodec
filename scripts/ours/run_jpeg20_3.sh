cd ../..

python train_ste.py --add_exp_version 1 --config configs/nsvf_bike/bike_codec_ste_jpeg20.txt \
                --ckpt log/tensorf_bike_VM/tensorf_bike_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 12000 \
                --save_every 4000 --vis_every 4000

python train_ste.py --add_exp_version 1 --config configs/nsvf_palace/palace_codec_ste_jpeg20.txt \
                --ckpt log/tensorf_palace_VM/tensorf_palace_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 12000 \
                --save_every 4000 --vis_every 4000

python train_ste.py --add_exp_version 1 --config configs/nsvf_robot/robot_codec_ste_jpeg20.txt \
                --ckpt log/tensorf_robot_VM/tensorf_robot_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 12000 \
                --save_every 4000 --vis_every 4000

python train_ste.py --add_exp_version 1 --config configs/nsvf_steamtrain/steamtrain_codec_ste_jpeg20.txt \
                --ckpt log/tensorf_steamtrain_VM/tensorf_steamtrain_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 12000 \
                --save_every 4000 --vis_every 4000