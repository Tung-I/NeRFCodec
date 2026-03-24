python train_ste_log_grad.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_jpeg65.txt \
                --ckpt log_2/tensorf_chair_VM/tensorf_chair_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project grad_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 10000 --vis_every 10000 \
                --expname grad_ste_chair_jpeg65
             

python train_ste_log_grad.py --add_exp_version 1 --config configs/nerf_drums/drums_codec_ste_jpeg65.txt \
                --ckpt log_2/tensorf_drums_VM/tensorf_drums_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project grad_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 10000 --vis_every 10000 \
                --expname grad_ste_drums_jpeg65
             

python train_ste_log_grad.py --add_exp_version 1 --config configs/nerf_ficus/ficus_codec_ste_jpeg65.txt \
                --ckpt log_2/tensorf_ficus_VM/tensorf_ficus_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project grad_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 10000 --vis_every 10000 \
                --expname grad_ste_ficus_jpeg65
             

python train_ste_log_grad.py --add_exp_version 1 --config configs/nerf_hotdog/hotdog_codec_ste_jpeg65.txt \
                --ckpt log_2/tensorf_hotdog_VM/tensorf_hotdog_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project grad_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 10000 --vis_every 10000 \
                --expname grad_ste_hotdog_jpeg65
             

python train_ste_log_grad.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg65.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project grad_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 10000 --vis_every 10000 \
                --expname grad_ste_lego_jpeg65
             

python train_ste_log_grad.py --add_exp_version 1 --config configs/nerf_materials/materials_codec_ste_jpeg65.txt \
                --ckpt log_2/tensorf_materials_VM/tensorf_materials_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project grad_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 10000 --vis_every 10000 \
                --expname grad_ste_materials_jpeg65
             

python train_ste_log_grad.py --add_exp_version 1 --config configs/nerf_mic/mic_codec_ste_jpeg65.txt \
                --ckpt log_2/tensorf_mic_VM/tensorf_mic_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project grad_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 10000 --vis_every 10000 \
                --expname grad_ste_mic_jpeg65
             

python train_ste_log_grad.py --add_exp_version 1 --config configs/nerf_ship/ship_codec_ste_jpeg65.txt \
                --ckpt log_2/tensorf_ship_VM/tensorf_ship_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project grad_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 10000 --vis_every 10000 \
                --expname grad_ste_ship_jpeg65
             
