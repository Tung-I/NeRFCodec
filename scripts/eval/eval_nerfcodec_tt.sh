cd ../..

python eval.py  \
    --dataset_name tankstemple \
    --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Truck \
    --system_ckpt log_2/tt_truck/truck_codec_384_system_39999.th \
    --ckpt log_2/tt_truck/truck_codec_384_compression_39999.th

python eval.py  \
    --dataset_name tankstemple \
    --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Caterpillar \
    --system_ckpt log_2/tt_caterpillar/caterpillar_codec_system_49999.th \
    --ckpt log_2/tt_caterpillar/caterpillar_codec_compression_49999.th


# python eval.py  \
#     --dataset_name tankstemple \
#     --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
#     --N_vis 5 \
#     --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Family \
#     --system_ckpt log_2/tt_family/family_codec_system_34999.th \
#     --ckpt log_2/tt_family/family_codec_compression_34999.th 