cd ../..


# python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_jpeg.txt \
#                 --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project jpeg_ste \
#                 --expname ste_nerf_chair_jpeg_qp35 \
#                 --den_quality 35 --app_quality 35\
#                 --n_iters 40000 --TV_weight_app 0.1 --refresh_k 32 \
#                 --save_every 10000 --vis_every 10000

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_jpeg.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --expname ste_nerf_chair_jpeg_qp50 \
                --den_quality 50 --app_quality 50\
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 32 \
                --save_every 10000 --vis_every 10000

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_jpeg.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --expname ste_nerf_chair_jpeg_qp65 \
                --den_quality 65 --app_quality 65\
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 32 \
                --save_every 10000 --vis_every 10000

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_jpeg.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --expname ste_nerf_chair_jpeg_qp80 \
                --den_quality 80 --app_quality 80\
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 32 \
                --save_every 10000 --vis_every 10000
