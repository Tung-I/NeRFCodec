cd ../..

python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/hotdog \
    --config configs/nerf_hotdog/hotdog_codec_ste_jpeg35.txt \
    --ckpt_dir log3/ours_nerf_hotdog_jpeg_qp35 \
    --train_iters 19999

python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/hotdog \
    --config configs/nerf_hotdog/hotdog_codec_ste_jpeg50.txt \
    --ckpt_dir log3/ours_nerf_hotdog_jpeg_qp50 \
    --train_iters 19999

python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/hotdog \
    --config configs/nerf_hotdog/hotdog_codec_ste_jpeg65.txt \
    --ckpt_dir log3/ours_nerf_hotdog_jpeg_qp65 \
    --train_iters 19999

python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/hotdog \
    --config configs/nerf_hotdog/hotdog_codec_ste_jpeg80.txt \
    --ckpt_dir log3/ours_nerf_hotdog_jpeg_qp80 \
    --train_iters 19999


python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/materials \
    --config configs/nerf_materials/materials_codec_ste_jpeg35.txt \
    --ckpt_dir log3/ours_nerf_materials_jpeg_qp35

python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/materials \
    --config configs/nerf_materials/materials_codec_ste_jpeg50.txt \
    --ckpt_dir log3/ours_nerf_materials_jpeg_qp50

python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/materials \
    --config configs/nerf_materials/materials_codec_ste_jpeg65.txt \
    --ckpt_dir log3/ours_nerf_materials_jpeg_qp65

python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/materials \
    --config configs/nerf_materials/materials_codec_ste_jpeg80.txt \
    --ckpt_dir log3/ours_nerf_materials_jpeg_qp80


python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/ship \
    --config configs/nerf_ship/ship_codec_ste_jpeg35.txt \
    --ckpt_dir log3/ours_nerf_ship_jpeg_qp35

python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/ship \
    --config configs/nerf_ship/ship_codec_ste_jpeg50.txt \
    --ckpt_dir log3/ours_nerf_ship_jpeg_qp50

python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/ship \
    --config configs/nerf_ship/ship_codec_ste_jpeg65.txt \
    --ckpt_dir log3/ours_nerf_ship_jpeg_qp65

python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/ship \
    --config configs/nerf_ship/ship_codec_ste_jpeg80.txt \
    --ckpt_dir log3/ours_nerf_ship_jpeg_qp80


python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/mic \
    --config configs/nerf_mic/mic_codec_ste_jpeg35.txt \
    --ckpt_dir log3/ours_nerf_mic_jpeg_qp35

python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/mic \
    --config configs/nerf_mic/mic_codec_ste_jpeg50.txt \
    --ckpt_dir log3/ours_nerf_mic_jpeg_qp50

python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/mic \
    --config configs/nerf_mic/mic_codec_ste_jpeg65.txt \
    --ckpt_dir log3/ours_nerf_mic_jpeg_qp65

python eval_ours_4in1.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/mic \
    --config configs/nerf_mic/mic_codec_ste_jpeg80.txt \
    --ckpt_dir log3/ours_nerf_mic_jpeg_qp80