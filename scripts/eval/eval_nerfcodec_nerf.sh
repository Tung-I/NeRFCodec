cd ../..

# python eval.py  \
#     --dataset_name blender \
#     --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
#     --N_vis 5 \
#     --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
#     --system_ckpt log_2/nerf_chair_384/chair_codec_384_system_19999.th \
#     --ckpt log_2/nerf_chair_384/chair_codec_384_compression_19999.th 

python eval.py  \
    --dataset_name blender \
    --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/drums \
    --system_ckpt log_2/nerf_drums_384/drums_codec_384_system_29999.th \
    --ckpt log_2/nerf_drums_384/drums_codec_384_compression_29999.th 


python eval.py  \
    --dataset_name blender \
    --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/ficus \
    --system_ckpt log_2/nerf_ficus_384/ficus_codec_384_system_19999.th \
    --ckpt log_2/nerf_ficus_384/ficus_codec_384_compression_19999.th 

python eval.py  \
    --dataset_name blender \
    --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/hotdog \
    --system_ckpt log_2/nerf_hotdog_384/hotdog_codec_384_system_4999.th \
    --ckpt log_2/nerf_hotdog_384/hotdog_codec_384_compression_4999.th 


python eval.py  \
    --dataset_name blender \
    --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/lego \
    --system_ckpt log_2/nerf_lego_384/lego_codec_384_system_64999.th \
    --ckpt log_2/nerf_lego_384/lego_codec_384_compression_64999.th 

python eval.py  \
    --dataset_name blender \
    --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/materials \
    --system_ckpt log_2/nerf_materials_384/materials_codec_384_system_24999.th \
    --ckpt log_2/nerf_materials_384/materials_codec_384_compression_24999.th 

python eval.py  \
    --dataset_name blender \
    --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/mic \
    --system_ckpt log_2/nerf_mic/mic_codec_system_29999.th \
    --ckpt log_2/nerf_mic/mic_codec_compression_29999.th 

python eval.py  \
    --dataset_name blender \
    --compression --compression_strategy adaptor_feat_coding --compress_before_volrend \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/ship \
    --system_ckpt log_2/nerf_ship/ship_codec_system_39999.th \
    --ckpt log_2/nerf_ship/ship_codec_compression_39999.th 