cd ../..

python eval_tensorf_3in1.py \
  --den_packing_mode flatten --den_quant_mode global --den_global_range -25 25 --den_r 4 --den_c 4 \
  --app_packing_mode flatten --app_quant_mode global --app_global_range -5 5 --app_r 6 --app_c 8 \
  --ckpt log_2/nerf_chair/chair_codec_compression_34999.th \
  --outdir log_2/nerf_chair/rec_ckpt_jpeg80 \
  --jpeg_quality 80 

python eval_tensorf_3in1.py \
  --den_packing_mode flatten --den_quant_mode global --den_global_range -25 25 --den_r 4 --den_c 4 \
  --app_packing_mode flatten --app_quant_mode global --app_global_range -5 5 --app_r 6 --app_c 8 \
  --ckpt log_2/nerf_chair/chair_codec_compression_34999.th \
  --outdir log_2/nerf_chair/rec_ckpt_jpeg20 \
  --jpeg_quality 20 



python eval_ckpt_direct.py \
  --dataset_name blender \
  --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
  --model_name TensorVMSplit \
  --ckpt log_2/tensorf_chair/tensorf_chair_VM.th \
  --N_vis 5
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
python eval_ckpt_direct.py \
  --dataset_name blender \
  --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
  --model_name TensorVMSplit \
  --ckpt log_2/nerf_chair/rec_ckpt_jpeg20/chair_codec_compression_34999_recon_jpeg20.pth \
  --N_vis 5


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