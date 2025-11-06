cd ../..
# python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_vp9.txt \
#                 --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project ours_nerf_chair \
#                 --n_iters 20000 \
#                 --save_every 2000 --vis_every 2000 \
#                 --refresh_k 32 \
#                 --expname ste_nerf_chair_vp9_qp28 \
#                 --den_vp9_qp 28 --app_vp9_qp 28\

# python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_vp9.txt \
#                 --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project ours_nerf_chair \
#                 --n_iters 20000 \
#                 --save_every 2000 --vis_every 2000 \
#                 --refresh_k 32 \
#                 --expname ste_nerf_chair_vp9_qp32 \
#                 --den_vp9_qp 32 --app_vp9_qp 32\

# python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_vp9.txt \
#                 --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project ours_nerf_chair \
#                 --n_iters 20000 \
#                 --save_every 2000 --vis_every 2000 \
#                 --refresh_k 32 \
#                 --expname ste_nerf_chair_vp9_qp36 \
#                 --den_vp9_qp 36 --app_vp9_qp 36\

# python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_vp9.txt \
#                 --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project ours_nerf_chair \
#                 --n_iters 20000 \
#                 --save_every 2000 --vis_every 2000 \
#                 --refresh_k 32 \
#                 --expname ste_nerf_chair_vp9_qp40 \
#                 --den_vp9_qp 40 --app_vp9_qp 40\

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_vp9.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ours_nerf_chair \
                --n_iters 20000 \
                --save_every 2000 --vis_every 2000 \
                --refresh_k 32 \
                --expname ste_nerf_chair_vp9_qp44 \
                --den_vp9_qp 44 --app_vp9_qp 44\