cd ../..
# python train_ste.py --add_exp_version 1 --config configs/drums_codec_ste_jpeg80.txt \
#                 --ckpt log/tensorf_drums_VM/tensorf_drums_VM.th\
#                 --compression --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project jpeg_ste \
#                 --save_every 5000 --vis_every 5000

# python train_ste.py --add_exp_version 1 --config configs/drums_codec_ste_jpeg65.txt \
#                 --ckpt log/tensorf_drums_VM/tensorf_drums_VM.th\
#                 --compression --batch_size 65536 \
#                 --codec_training \
#                 --lr_decay_target_ratio 1 \
#                 --wandb_project jpeg_ste \
#                 --save_every 5000 --vis_every 5000

python train_ste.py --add_exp_version 1 --config configs/drums_codec_ste_jpeg50.txt \
                --ckpt log/tensorf_drums_VM/tensorf_drums_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --save_every 5000 --vis_every 5000


python train_ste.py --add_exp_version 1 --config configs/drums_codec_ste_jpeg35.txt \
                --ckpt log/tensorf_drums_VM/tensorf_drums_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 \
                --wandb_project jpeg_ste \
                --save_every 5000 --vis_every 5000


