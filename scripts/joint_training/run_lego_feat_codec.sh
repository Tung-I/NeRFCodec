python train.py --add_exp_version 1 --expname lego_codec \
                --config configs/lego_codec.txt --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --warm_up_ckpt log/lego_codec/version_000 \
                --save_every 5000 --vis_every 5000

# python train.py --add_exp_version 1 --expname lego_codec \
#                 --config configs/lego_codec.txt --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up --warm_up_ckpt log/only_adaptor/version_003


python train_ste.py --add_exp_version 1 --config configs/lego_codec_ste_jpeg80.txt \
                --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --batch_size 65536 \
                --codec_training \
                --lr_decay_target_ratio 1 


