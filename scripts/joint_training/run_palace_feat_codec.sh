python train.py --add_exp_version 1 --expname palace_codec \
                --config configs/palace_codec.txt --ckpt log/tensorf_palace_VM/tensorf_palace_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000

python train.py --add_exp_version 1 --expname bike_codec \
                --config configs/bike_codec.txt --ckpt log/tensorf_bike_VM/tensorf_bike_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000

python train.py --add_exp_version 1 --expname lifestyle_codec \
                --config configs/lifestyle_codec.txt --ckpt log/tensorf_lifestyle_VM/tensorf_lifestyle_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000

python train.py --add_exp_version 1 --expname robot_codec \
                --config configs/robot_codec.txt --ckpt log/tensorf_robot_VM/tensorf_robot_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000

python train.py --add_exp_version 1 --expname spaceship_codec \
                --config configs/spaceship_codec.txt --ckpt log/tensorf_spaceship_VM/tensorf_spaceship_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000

python train.py --add_exp_version 1 --expname steamtrain_codec \
                --config configs/steamtrain_codec.txt --ckpt log/tensorf_steamtrain_VM/tensorf_steamtrain_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000


python train.py --add_exp_version 1 --expname toad_codec \
                --config configs/toad_codec.txt --ckpt log/tensorf_toad_VM/tensorf_toad_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000


python train.py --add_exp_version 1 --expname wineholder_codec \
                --config configs/wineholder_codec.txt --ckpt log/tensorf_wineholder_VM/tensorf_wineholder_VM.th\
                --compression --render_test 1 --batch_size 65536 \
                --codec_training --compression_strategy adaptor_feat_coding \
                --lr_feat_codec 2e-4 --lr_aux 1e-3 \
                --fix_decoder_prior \
                --compress_before_volrend \
                --rate_penalty \
                --lr_decay_target_ratio 1 \
                --warm_up \
                --save_every 5000 --vis_every 5000