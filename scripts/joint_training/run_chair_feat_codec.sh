# export CUDA_VISIBLE_DEVICES=7
### per scene joint training
# now triplane & channel adaptor & reserved coding network are trained together
python train.py --add_exp_version 1 --expname chair_codec \
                --config configs/chair_codec.txt --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up

python train.py --add_exp_version 1 --expname chair_codec \
                --config configs/chair_codec.txt --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up --warm_up_ckpt log/only_adaptor/version_002
