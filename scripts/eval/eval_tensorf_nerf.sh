cd ../..

# python eval_tensorf.py \
#     --dataset_name blender \
#     --N_vis 5 \
#     --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
#     --config configs/nerf_chair/chair.txt \
#     --ckpt_dir log/tensorf_chair_VM

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/drums \
    --config configs/nerf_drums/drums.txt \
    --ckpt_dir log/tensorf_drums_VM

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/ficus \
    --config configs/nerf_ficus/ficus.txt \
    --ckpt_dir log/tensorf_ficus_VM

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/hotdog \
    --config configs/nerf_hotdog/hotdog.txt \
    --ckpt_dir log/tensorf_hotdog_VM

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/legor \
    --config configs/nerf_legor/legor.txt \
    --ckpt_dir log/tensorf_legor_VM

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/materials \
    --config configs/nerf_materials/materials.txt \
    --ckpt_dir log/tensorf_materials_VM

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/mic \
    --config configs/nerf_mic/mic.txt \
    --ckpt_dir log/tensorf_mic_VM

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/ship \
    --config configs/nerf_ship/ship.txt \
    --ckpt_dir log/tensorf_ship_VM