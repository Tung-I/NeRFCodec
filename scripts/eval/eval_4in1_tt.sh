cd ../..

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Barn \
    --config configs/tt_barn/barn_codec_ste_jpeg50.txt \
    --ckpt_dir log3/ours_tt_barn_jpeg_qp50 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Barn \
    --config configs/tt_barn/barn_codec_ste_jpeg35.txt \
    --ckpt_dir log3/ours_tt_barn_jpeg_qp35 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Barn \
    --config configs/tt_barn/barn_codec_ste_jpeg20.txt \
    --ckpt_dir log3/ours_tt_barn_jpeg_qp20 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Barn \
    --config configs/tt_barn/barn_codec_ste_jpeg80.txt \
    --ckpt_dir log3/ours_tt_barn_jpeg_qp80 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Caterpillar \
    --config configs/tt_caterpillar/caterpillar_codec_ste_jpeg50.txt \
    --ckpt_dir log3/ours_tt_caterpillar_jpeg_qp50 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Caterpillar \
    --config configs/tt_caterpillar/caterpillar_codec_ste_jpeg35.txt \
    --ckpt_dir log3/ours_tt_caterpillar_jpeg_qp35 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Caterpillar \
    --config configs/tt_caterpillar/caterpillar_codec_ste_jpeg20.txt \
    --ckpt_dir log3/ours_tt_caterpillar_jpeg_qp20 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Caterpillar \
    --config configs/tt_caterpillar/caterpillar_codec_ste_jpeg80.txt \
    --ckpt_dir log3/ours_tt_caterpillar_jpeg_qp80 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Family \
    --config configs/tt_family/family_codec_ste_jpeg50.txt \
    --ckpt_dir log3/ours_tt_family_jpeg_qp50 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Family \
    --config configs/tt_family/family_codec_ste_jpeg35.txt \
    --ckpt_dir log3/ours_tt_family_jpeg_qp35 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Family \
    --config configs/tt_family/family_codec_ste_jpeg20.txt \
    --ckpt_dir log3/ours_tt_family_jpeg_qp20 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Family \
    --config configs/tt_family/family_codec_ste_jpeg80.txt \
    --ckpt_dir log3/ours_tt_family_jpeg_qp80 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Truck \
    --config configs/tt_truck/truck_codec_ste_jpeg50.txt \
    --ckpt_dir log3/ours_tt_truck_jpeg_qp50 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Truck \
    --config configs/tt_truck/truck_codec_ste_jpeg35.txt \
    --ckpt_dir log3/ours_tt_truck_jpeg_qp35 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Truck \
    --config configs/tt_truck/truck_codec_ste_jpeg20.txt \
    --ckpt_dir log3/ours_tt_truck_jpeg_qp20 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Truck \
    --config configs/tt_truck/truck_codec_ste_jpeg80.txt \
    --ckpt_dir log3/ours_tt_truck_jpeg_qp80 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Ignatius \
    --config configs/tt_ignatius/ignatius_codec_ste_jpeg50.txt \
    --ckpt_dir log3/ours_tt_ignatius_jpeg_qp50 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Ignatius \
    --config configs/tt_ignatius/ignatius_codec_ste_jpeg35.txt \
    --ckpt_dir log3/ours_tt_ignatius_jpeg_qp35 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Ignatius \
    --config configs/tt_ignatius/ignatius_codec_ste_jpeg20.txt \
    --ckpt_dir log3/ours_tt_ignatius_jpeg_qp20 \
    --downsample_train 2

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 8 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Ignatius \
    --config configs/tt_ignatius/ignatius_codec_ste_jpeg80.txt \
    --ckpt_dir log3/ours_tt_ignatius_jpeg_qp80 \
    --downsample_train 2