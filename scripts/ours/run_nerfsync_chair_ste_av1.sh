cd ../..
# python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_av1.txt \
#                 --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project ours_nerf_chair \
#                 --n_iters 20000 \
#                 --save_every 2000 --vis_every 2000 \
#                 --refresh_k 32 \
#                 --expname ste_nerf_chair_av1_qp44 \
#                 --den_av1_qp 44 --app_av1_qp 44\

# python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_av1.txt \
#                 --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project ours_nerf_chair \
#                 --n_iters 20000 \
#                 --save_every 2000 --vis_every 2000 \
#                 --refresh_k 32 \
#                 --expname ste_nerf_chair_av1_qp20 \
#                 --den_av1_qp 20 --app_av1_qp 20\

# python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_av1.txt \
#                 --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project ours_nerf_chair \
#                 --n_iters 20000 \
#                 --save_every 2000 --vis_every 2000 \
#                 --refresh_k 32 \
#                 --expname ste_nerf_chair_av1_qp26 \
#                 --den_av1_qp 26 --app_av1_qp 26\

# python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_av1.txt \
#                 --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project ours_nerf_chair \
#                 --n_iters 20000 \
#                 --save_every 2000 --vis_every 2000 \
#                 --refresh_k 32 \
#                 --expname ste_nerf_chair_av1_qp32 \
#                 --den_av1_qp 32 --app_av1_qp 32\

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_av1.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ours_nerf_chair \
                --n_iters 20000 \
                --save_every 2000 --vis_every 2000 \
                --refresh_k 32 \
                --expname ste_nerf_chair_av1_qp38 \
                --den_av1_qp 38 --app_av1_qp 38\
