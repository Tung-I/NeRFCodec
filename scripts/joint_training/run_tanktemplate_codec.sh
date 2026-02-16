python train.py --add_exp_version 1 --expname truck_codec \
                --config configs/truck_codec.txt --ckpt log/tensorf_truck_VM/tensorf_truck_VM.th\
                --compression --render_test 1 --batch_size 16384 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --warm_up_ckpt log/truck_codec/version_000 \
                --save_every 5000 --vis_every 5000 \
                --downsample_train 2

python train.py --add_exp_version 1 --expname truck_codec_384 \
                --config configs/truck_codec_384.txt --ckpt log/tensorf_truck_VM_old/tensorf_truck_VM.th\
                --compression --render_test 1 --batch_size 16384 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --warm_up_ckpt log/truck_codec/version_000 \
                --save_every 5000 --vis_every 5000 \
                --downsample_train 2

# python train.py --add_exp_version 1 --expname truck_codec_384 \
#                 --config configs/truck_codec_384.txt --ckpt log/tensorf_truck_VM/tensorf_truck_VM.th\
#                 --compression --render_test 1 --batch_size 16384 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --save_every 5000 --vis_every 5000 \
#                 --downsample_train 2

# python train.py --add_exp_version 1 --expname family_codec \
#                 --config configs/family_codec.txt --ckpt log/tensorf_family_VM/tensorf_family_VM.th\
#                 --compression --render_test 1 --batch_size 16384 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --warm_up_ckpt log/family_codec/version_000 \
#                 --save_every 5000 --vis_every 5000 \
#                 --downsample_train 2

python train.py --add_exp_version 1 --expname family_codec \
                --config configs/family_codec.txt \
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --save_every 5000 --vis_every 5000 \
                --downsample_train 2 \
                --resume_optim \
                --resume_system_ckpt log/family_codec/version_000/family_codec_system_24999.th \
                --extra_iters 60000



python train.py --add_exp_version 1 --expname barn_codec \
                --config configs/barn_codec.txt --ckpt log/tensorf_barn_VM/tensorf_barn_VM.th\
                --compression --render_test 1 --batch_size 16384 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --warm_up_ckpt log/barn_codec/version_000 \
                --save_every 5000 --vis_every 5000 \
                --downsample_train 2

python train.py --add_exp_version 1 --expname caterpillar_codec \
                --config configs/caterpillar_codec.txt --ckpt log/tensorf_caterpillar_VM/tensorf_caterpillar_VM.th\
                --compression --render_test 1 --batch_size 16384 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000 \
                --warm_up_ckpt log/caterpillar_codec/version_000 \
                --downsample_train 2

python train.py --add_exp_version 1 --expname ignatius_codec \
                --config configs/tt_ignatius/ignatius_384.txt --ckpt log/tensorf_ignatius_VM/tensorf_ignatius_VM.th\
                --compression --render_test 1 --batch_size 16384 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000 \
                --downsample_train 2 \
                --warm_up_ckpt log/ignatius_codec/version_000 
                

######################################################3

