cd ../..

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/chair \
    --config configs/nerf_chair/chair.txt \
    --ckpt_dir log/tensorf_chair_VM

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/drums \
    --config configs/nerf_drums/drums.txt \
    --ckpt_dir log/tensorf_drums_VM

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/ficus \
    --config configs/nerf_ficus/ficus.txt \
    --ckpt_dir log/tensorf_ficus_VM

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/hotdog \
    --config configs/nerf_hotdog/hotdog.txt \
    --ckpt_dir log/tensorf_hotdog_VM

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/lego \
    --config configs/nerf_lego/lego.txt \
    --ckpt_dir log/tensorf_lego_VM

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/materials \
    --config configs/nerf_materials/materials.txt \
    --ckpt_dir log/tensorf_materials_VM

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/mic \
    --config configs/nerf_mic/mic.txt \
    --ckpt_dir log/tensorf_mic_VM

python eval_tensorf.py \
    --dataset_name blender \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/nerf_synthetic/ship \
    --config configs/nerf_ship/ship.txt \
    --ckpt_dir log/tensorf_ship_VM

python eval_tensorf.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Barn \
    --config configs/barn.txt \
    --ckpt_dir log/tensorf_barn_VM

python eval_tensorf.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Family \
    --config configs/family.txt \
    --ckpt_dir log/tensorf_family_VM

python eval_tensorf.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Caterpillar \
    --config configs/caterpillar.txt \
    --ckpt_dir log/tensorf_caterpillar_VM

python eval_tensorf.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Ignatius \
    --config configs/ignatius.txt \
    --ckpt_dir log/tensorf_ignatius_VM

python eval_tensorf.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Truck \
    --config configs/truck.txt \
    --ckpt_dir log/tensorf_truck_VM