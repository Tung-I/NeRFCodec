cd ../..

python train_ste.py --add_exp_version 1 --config configs/tt_barn/barn_codec_ste_jpeg80.txt \
                --ckpt log/tensorf_barn_VM/tensorf_barn_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 30000 --TV_weight_app 0.01 --refresh_k 64 \
                --save_every 10000 --vis_every 10000 \
                --downsample_train 2

python train_ste.py --add_exp_version 1 --config configs/tt_caterpillar/caterpillar_codec_ste_jpeg80.txt \
                --ckpt log/tensorf_caterpillar_VM/tensorf_caterpillar_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 30000 --TV_weight_app 0.01 --refresh_k 64 \
                --save_every 10000 --vis_every 10000 \
                --downsample_train 2

python train_ste.py --add_exp_version 1 --config configs/tt_ignatius/ignatius_codec_ste_jpeg80.txt \
                --ckpt log/tensorf_ignatius_VM/tensorf_ignatius_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 30000 --TV_weight_app 0.01 --refresh_k 64 \
                --save_every 10000 --vis_every 10000 \
                --downsample_train 2

python train_ste.py --add_exp_version 1 --config configs/tt_truck/truck_codec_ste_jpeg80.txt \
                --ckpt log/tensorf_truck_VM/tensorf_truck_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --n_iters 30000 --TV_weight_app 0.01 --refresh_k 64 \
                --save_every 10000 --vis_every 10000 \
                --downsample_train 2

# python train_ste.py --add_exp_version 1 --config configs/tt_barn/barn_codec_ste_jpeg50.txt \
#                 --ckpt log/tensorf_barn_VM/tensorf_barn_VM.th\
#                 --compression --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project jpeg_ste \
#                 --n_iters 30000 --TV_weight_app 0.1 --refresh_k 32 \
#                 --save_every 10000 --vis_every 10000 \
#                 --downsample_train 2

# python train_ste.py --add_exp_version 1 --config configs/tt_barn/barn_codec_ste_jpeg35.txt \
#                 --ckpt log/tensorf_barn_VM/tensorf_barn_VM.th\
#                 --compression --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project jpeg_ste \
#                 --n_iters 30000 --TV_weight_app 0.1 --refresh_k 32 \
#                 --save_every 10000 --vis_every 10000 \
#                 --downsample_train 2

# python train_ste.py --add_exp_version 1 --config configs/tt_barn/barn_codec_ste_jpeg20.txt \
#                 --ckpt log/tensorf_barn_VM/tensorf_barn_VM.th\
#                 --compression --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project jpeg_ste \
#                 --n_iters 30000 --TV_weight_app 0.1 --refresh_k 32 \
#                 --save_every 10000 --vis_every 10000 \
#                 --downsample_train 2

# python train_ste.py --add_exp_version 1 --config configs/tt_barn/barn_codec_ste_jpeg10.txt \
#                 --ckpt log/tensorf_barn_VM/tensorf_barn_VM.th\
#                 --compression --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project jpeg_ste \
#                 --n_iters 30000 --TV_weight_app 0.1 --refresh_k 32 \
#                 --save_every 10000 --vis_every 10000 \
#                 --downsample_train 2
