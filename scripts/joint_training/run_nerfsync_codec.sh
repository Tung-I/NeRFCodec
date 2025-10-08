python noise_train.py --add_exp_version 1 --expname lego_noise_lowpass_0 \
                --config configs/lego_noise.txt --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --lr_decay_target_ratio 1 \
                --save_every 100 --vis_every 100 \
                --codec_noise --codec_noise_mode lowpass_q --codec_noise_level 0

python noise_train.py --add_exp_version 1 --expname lego_noise_lowpass_1 \
                --config configs/lego_noise.txt --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --lr_decay_target_ratio 1 \
                --save_every 100 --vis_every 100 \
                --codec_noise --codec_noise_mode lowpass_q --codec_noise_level 1

python noise_train.py --add_exp_version 1 --expname lego_noise_blocky_0 \
                --config configs/lego_noise.txt --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --lr_decay_target_ratio 1 \
                --save_every 100 --vis_every 100 \
                --codec_noise --codec_noise_mode blocky --codec_noise_level 0

python noise_train.py --add_exp_version 1 --expname lego_noise_blocky_1 \
                --config configs/lego_noise.txt --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --lr_decay_target_ratio 1 \
                --save_every 100 --vis_every 100 \
                --codec_noise --codec_noise_mode blocky --codec_noise_level 1

python noise_train.py --add_exp_version 1 --expname lego_noise_gaussian_0 \
                --config configs/lego_noise.txt --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --lr_decay_target_ratio 1 \
                --save_every 100 --vis_every 100 \
                --codec_noise --codec_noise_mode gaussian --codec_noise_level 0

python noise_train.py --add_exp_version 1 --expname lego_noise_gaussian_1 \
                --config configs/lego_noise.txt --ckpt log/tensorf_lego_VM/tensorf_lego_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --lr_decay_target_ratio 1 \
                --save_every 100 --vis_every 100 \
                --codec_noise --codec_noise_mode gaussian --codec_noise_level 1

#####################################################################
python train.py --add_exp_version 1 --expname chair_codec_384 \
                --config configs/chair_codec_384.txt \
                --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000 

python train.py --add_exp_version 1 --expname drums_codec_384 \
                --config configs/drums_codec_384.txt \
                --ckpt log/tensorf_drums_VM/tensorf_drums_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000 


python train.py --add_exp_version 1 --expname ficus_codec_384 \
                --config configs/ficus_codec_384.txt \
                --ckpt log/tensorf_ficus_VM/tensorf_ficus_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000 

# python train.py --add_exp_version 1 --expname hotdog_codec_384 \
#                 --config configs/hotdog_codec_384.txt \
#                 --ckpt log/tensorf_hotdog_VM/tensorf_hotdog_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --save_every 5000 --vis_every 5000 

python train.py --add_exp_version 1 --expname hotdog_codec_384 \
                --config configs/hotdog_codec_384.txt \
                --ckpt log/tensorf_hotdog_VM/tensorf_hotdog_VM.th\
                --compression --render_test 1 --batch_size 16384  \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --warm_up_ckpt log/hotdog_codec_384/version_001 \
                --save_every 5000 --vis_every 5000 

python train.py --add_exp_version 1 --expname mic_codec_384 \
                --config configs/mic_codec_384.txt \
                --ckpt log/tensorf_mic_VM/tensorf_mic_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000 


python train.py --add_exp_version 1 --expname materials_codec_384 \
                --config configs/materials_codec_384.txt \
                --ckpt log/tensorf_materials_VM/tensorf_materials_VM.th\
                --compression --render_test 1 --batch_size 16384 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --warm_up_ckpt log/materials_codec_384/version_001 \
                --save_every 5000 --vis_every 5000 


    


#####################################################################
python train.py --add_exp_version 1 --expname chair_codec \
                --config configs/chair_codec.txt \
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --save_every 5000 --vis_every 5000 \
                --resume_optim \
                --resume_system_ckpt log/chair_codec/version_001/chair_codec_system_29999.th \
                --extra_iters 60000

python train.py --add_exp_version 1 --expname drums_codec \
                --config configs/drums_codec.txt \
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --save_every 5000 --vis_every 5000 \
                --resume_optim \
                --resume_system_ckpt log/drums_codec/version_001/drums_codec_system_34999.th \
                --extra_iters 60000

# python train.py --add_exp_version 1 --expname ficus_codec \
#                 --config configs/ficus_codec.txt \
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --save_every 5000 --vis_every 5000 \
#                 --resume_optim \
#                 --resume_system_ckpt log/ficus_codec/version_001/ficus_codec_system_34999.th \
#                 --extra_iters 60000

python train.py --add_exp_version 1 --expname ficus_codec \
                --config configs/ficus_codec.txt --ckpt log/tensorf_ficus_VM/tensorf_ficus_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000

# python train.py --add_exp_version 1 --expname hotdog_codec \
#                 --config configs/hotdog_codec.txt \
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --save_every 5000 --vis_every 5000 \
#                 --resume_optim \
#                 --resume_system_ckpt log/hotdog_codec/version_001/hotdog_codec_system_34999.th \
#                 --extra_iters 60000

python train.py --add_exp_version 1 --expname hotdog_codec \
                --config configs/hotdog_codec.txt --ckpt log/tensorf_hotdog_VM/tensorf_hotdog_VM.th\
                --compression --render_test 1 --batch_size 32768 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --warm_up_ckpt log/hotdog_codec/version_001 \
                --save_every 5000 --vis_every 5000 

python train.py --add_exp_version 1 --expname materials_codec \
                --config configs/materials_codec.txt --ckpt log/tensorf_materials_VM/tensorf_materials_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --warm_up_ckpt log/materials_codec/version_002 \
                --save_every 5000 --vis_every 5000 

python train.py --add_exp_version 1 --expname mic_codec \
                --config configs/mic_codec.txt \
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --save_every 5000 --vis_every 5000 \
                --resume_optim \
                --resume_system_ckpt log/mic_codec/version_001/mic_codec_system_34999.th \
                --extra_iters 60000

python train.py --add_exp_version 1 --expname ship_codec \
                --config configs/ship_codec.txt --ckpt log/tensorf_ship_VM/tensorf_ship_VM.th\
                --compression --render_test 1 --batch_size 32768 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --warm_up_ckpt log/ship_codec/version_001 \
                --save_every 5000 --vis_every 5000 

# python train.py --add_exp_version 1 --expname chair_codec \
#                 --config configs/chair_codec.txt --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --save_every 5000 --vis_every 5000

# python train.py --add_exp_version 1 --expname drums_codec \
#                 --config configs/drums_codec.txt --ckpt log/tensorf_drums_VM/tensorf_drums_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --save_every 5000 --vis_every 5000

# python train.py --add_exp_version 1 --expname ficus_codec \
#                 --config configs/ficus_codec.txt --ckpt log/tensorf_ficus_VM/tensorf_ficus_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --save_every 5000 --vis_every 5000


# python train.py --add_exp_version 1 --expname hotdog_codec \
#                 --config configs/hotdog_codec.txt --ckpt log/tensorf_hotdog_VM/tensorf_hotdog_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --save_every 5000 --vis_every 5000


# python train.py --add_exp_version 1 --expname materials_codec \
#                 --config configs/materials_codec.txt --ckpt log/tensorf_materials_VM/tensorf_materials_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --save_every 5000 --vis_every 5000


# python train.py --add_exp_version 1 --expname mic_codec \
#                 --config configs/mic_codec.txt --ckpt log/tensorf_mic_VM/tensorf_mic_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --save_every 5000 --vis_every 5000

# python train.py --add_exp_version 1 --expname ship_codec \
#                 --config configs/ship_codec.txt --ckpt log/tensorf_ship_VM/tensorf_ship_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --save_every 5000 --vis_every 5000



# python train.py --add_exp_version 1 --expname chair_codec \
#                 --config configs/chair_codec.txt --ckpt log/tensorf_chair_VM/tensorf_chair_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --warm_up_ckpt log/chair_codec/version_000 \
#                 --save_every 5000 --vis_every 5000

# python train.py --add_exp_version 1 --expname drums_codec \
#                 --config configs/drums_codec.txt --ckpt log/tensorf_drums_VM/tensorf_drums_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --warm_up_ckpt log/drums_codec/version_000 \
#                 --save_every 5000 --vis_every 5000

# python train.py --add_exp_version 1 --expname ficus_codec \
#                 --config configs/ficus_codec.txt --ckpt log/tensorf_ficus_VM/tensorf_ficus_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --warm_up_ckpt log/ficus_codec/version_000 \
#                 --save_every 5000 --vis_every 5000


# python train.py --add_exp_version 1 --expname hotdog_codec \
#                 --config configs/hotdog_codec.txt --ckpt log/tensorf_hotdog_VM/tensorf_hotdog_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --warm_up_ckpt log/hotdog_codec/version_000 \
#                 --save_every 5000 --vis_every 5000


# python train.py --add_exp_version 1 --expname materials_codec \
#                 --config configs/materials_codec.txt --ckpt log/tensorf_materials_VM/tensorf_materials_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --warm_up_ckpt log/materials_codec/version_000 \
#                 --save_every 5000 --vis_every 5000


# python train.py --add_exp_version 1 --expname mic_codec \
#                 --config configs/mic_codec.txt --ckpt log/tensorf_mic_VM/tensorf_mic_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --warm_up_ckpt log/mic_codec/version_000 \
#                 --save_every 5000 --vis_every 5000

# python train.py --add_exp_version 1 --expname ship_codec \
#                 --config configs/ship_codec.txt --ckpt log/tensorf_ship_VM/tensorf_ship_VM.th\
#                 --compression --render_test 1 --batch_size 65536 \
#                 --codec_training --compression_strategy adaptor_feat_coding \
#                 --lr_feat_codec 2e-4 --lr_aux 1e-3 \
#                 --fix_decoder_prior \
#                 --compress_before_volrend \
#                 --rate_penalty \
#                 --lr_decay_target_ratio 1 \
#                 --warm_up \
#                 --warm_up_ckpt log/ship_codec/version_000 \
#                 --save_every 5000 --vis_every 5000