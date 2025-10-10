cd ../..

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_jpeg20.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project nerf_chair_ste \
                --n_iters 20000 \
                --save_every 2000 --vis_every 2000  

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_jpeg35.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project nerf_chair_ste \
                --n_iters 16000 \
                --save_every 2000 --vis_every 2000  

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_jpeg65.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project nerf_chair_ste \
                --n_iters 8000 \
                --save_every 2000 --vis_every 2000

python train_ste.py --add_exp_version 1 --config configs/nerf_chair/chair_codec_ste_jpeg50.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project nerf_chair_ste \
                --n_iters 8000 \
                --save_every 2000 --vis_every 2000
   

 
