python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg65.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 5000 --vis_every 5000 \
                --expname ste_lego_jpeg65 

python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg20.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 5000 --vis_every 5000 \
                --expname ste_lego_jpeg20 

python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg35.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 5000 --vis_every 5000 \
                --expname ste_lego_jpeg35 

python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg50.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 5000 --vis_every 5000 \
                --expname ste_lego_jpeg50 


python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg50.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 5000 --vis_every 5000 \
                --expname mste_lego_jpeg50 --grad_surrogate_mode mste_std --grad_surrogate_std_eps 1e-8

python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg35.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 5000 --vis_every 5000 \
                --expname mste_lego_jpeg35 --grad_surrogate_mode mste_std --grad_surrogate_std_eps 1e-8

python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg65.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 5000 --vis_every 5000 \
                --expname mste_lego_jpeg65 --grad_surrogate_mode mste_std --grad_surrogate_std_eps 1e-8



python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg20.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 20000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 5000 --vis_every 5000 \
                --expname mste_lego_jpeg20 --grad_surrogate_mode mste_std --grad_surrogate_std_eps 1e-8


python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg65.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 50000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 5000 --vis_every 5000 \
                --expname spsa2_lego_jpeg65 --grad_surrogate_mode spsa --spsa_n_samples 2 --spsa_gate_on_cache_refresh 1 

python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg20.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 50000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 5000 --vis_every 5000 \
                --expname spsa2_lego_jpeg20 --grad_surrogate_mode spsa --spsa_n_samples 2 --spsa_gate_on_cache_refresh 1 

python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg50.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 50000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 5000 --vis_every 5000 \
                --expname spsa2_lego_jpeg50 --grad_surrogate_mode spsa --spsa_n_samples 2 --spsa_gate_on_cache_refresh 1 




python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg65.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 8 \
                --save_every 5000 --vis_every 5000 \
                --expname spsa8_lego_jpeg65 --grad_surrogate_mode spsa --spsa_n_samples 8 --spsa_gate_on_cache_refresh 1  

python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg65.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 8 \
                --save_every 5000 --vis_every 5000 \
                --expname spsa4_lego_jpeg65 --grad_surrogate_mode spsa --spsa_n_samples 4 --spsa_gate_on_cache_refresh 1

python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg65.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 8 \
                --save_every 5000 --vis_every 5000 \
                --expname spsa2_lego_jpeg65 --grad_surrogate_mode spsa --spsa_n_samples 2 --spsa_gate_on_cache_refresh 1

python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg20.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 8 \
                --save_every 5000 --vis_every 5000 \
                --expname spsa2_lego_jpeg20 --grad_surrogate_mode spsa --spsa_n_samples 2 --spsa_gate_on_cache_refresh 1


python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg20.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 8 \
                --save_every 5000 --vis_every 5000 \
                --expname spsa4_lego_jpeg20 --grad_surrogate_mode spsa --spsa_n_samples 4 --spsa_gate_on_cache_refresh 1

python train_ste.py --add_exp_version 1 --config configs/nerf_lego/lego_codec_ste_jpeg20.txt \
                --ckpt log_2/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 8 \
                --save_every 5000 --vis_every 5000 \
                --expname spsa8_lego_jpeg20 --grad_surrogate_mode spsa --spsa_n_samples 8 --spsa_gate_on_cache_refresh 1












python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_ste_jpeg65.txt \
                --ckpt log_2/tensorf_chair_VM/tensorf_chair_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 32 \
                --save_every 10000 --vis_every 10000 

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_ste_jpeg20.txt \
                --ckpt log_2/tensorf_chair_VM/tensorf_chair_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 32 \
                --save_every 10000 --vis_every 10000 

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_mste_jpeg65.txt \
                --ckpt log_2/tensorf_chair_VM/tensorf_chair_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 32 \
                --save_every 10000 --vis_every 10000 

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_mste_jpeg20.txt \
                --ckpt log_2/tensorf_chair_VM/tensorf_chair_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 32 \
                --save_every 10000 --vis_every 10000 

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_spsa_jpep65.txt \
                --ckpt log_2/tensorf_chair_VM/tensorf_chair_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project ablate_ste \
                --n_iters 30000 --TV_weight_app 0.1 --refresh_k 1 \
                --save_every 10000 --vis_every 10000 