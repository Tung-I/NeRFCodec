cd ../..

# python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_hevc.txt \
#                 --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project ours_nerf_chair \
#                 --n_iters 20000 \
#                 --save_every 2000 --vis_every 2000 \
#                 --refresh_k 32 \
#                 --expname ste_nerf_chair_hevc_qp28 \
#                 --den_hevc_qp 28 --app_hevc_qp 28\

# python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_hevc.txt \
#                 --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project ours_nerf_chair \
#                 --n_iters 20000 \
#                 --save_every 2000 --vis_every 2000 \
#                 --refresh_k 32 \
#                 --expname ste_nerf_chair_hevc_qp32 \
#                 --den_hevc_qp 32 --app_hevc_qp 32\

# python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_hevc.txt \
#                 --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project ours_nerf_chair \
#                 --n_iters 20000 \
#                 --save_every 2000 --vis_every 2000 \
#                 --refresh_k 32 \
#                 --expname ste_nerf_chair_hevc_qp36 \
#                 --den_hevc_qp 36 --app_hevc_qp 36\

# python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_hevc.txt \
#                 --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project ours_nerf_chair \
#                 --n_iters 20000 \
#                 --save_every 2000 --vis_every 2000 \
#                 --refresh_k 32 \
#                 --expname ste_nerf_chair_hevc_qp40 \
#                 --den_hevc_qp 40 --app_hevc_qp 40\

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_hevc.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ours_nerf_chair \
                --n_iters 20000 \
                --save_every 2000 --vis_every 2000 \
                --refresh_k 32 \
                --expname ste_nerf_chair_hevc_qp44 \
                --den_hevc_qp 44 --app_hevc_qp 44\

