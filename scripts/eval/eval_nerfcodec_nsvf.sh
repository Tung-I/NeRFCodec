cd ../..

# python eval.py  \
#     --dataset_name nsvf \
#     --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
#     --N_vis 5 \
#     --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/Synthetic_NSVF/Bike \
#     --system_ckpt log_2/nsvf_bike_384/bike_codec_384_system_24999.th \
#     --ckpt log_2/nsvf_bike_384/bike_codec_384_compression_24999.th 

python eval.py  \
    --dataset_name nsvf \
    --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/Synthetic_NSVF/Palace \
    --system_ckpt log_2/nsvf_palace_384/palace_codec_384_system_24999.th \
    --ckpt log_2/nsvf_palace_384/palace_codec_384_compression_24999.th 

python eval.py  \
    --dataset_name nsvf \
    --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/Synthetic_NSVF/Robot \
    --system_ckpt log_2/nsvf_robot_384/robot_codec_384_system_29999.th \
    --ckpt log_2/nsvf_robot_384/robot_codec_384_compression_29999.th 

python eval.py  \
    --dataset_name nsvf \
    --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/Synthetic_NSVF/Steamtrain \
    --system_ckpt log_2/nsvf_steamtrain_384/steamtrain_codec_384_system_14999.th \
    --ckpt log_2/nsvf_steamtrain_384/steamtrain_codec_384_compression_14999.th 