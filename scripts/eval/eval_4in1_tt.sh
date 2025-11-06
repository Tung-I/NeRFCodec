cd ../..

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Family \
    --config configs/tt_family/family_codec_ste_jpeg20.txt \
    --ckpt_dir log3/ours_tt_family_jpeg_qp20

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Family \
    --config configs/tt_family/family_codec_ste_jpeg35.txt \
    --ckpt_dir log3/ours_tt_family_jpeg_qp35

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Family \
    --config configs/tt_family/family_codec_ste_jpeg50.txt \
    --ckpt_dir log3/ours_tt_family_jpeg_qp50

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Family \
    --config configs/tt_family/family_codec_ste_jpeg65.txt \
    --ckpt_dir log3/ours_tt_family_jpeg_qp65

python eval_ours_4in1.py \
    --dataset_name tankstemple \
    --N_vis 5 \
    --datadir /work/pi_rsitaram_umass_edu/tungi/datasets/TanksAndTemples/Family \
    --config configs/tt_family/family_codec_ste_jpeg80.txt \
    --ckpt_dir log3/ours_tt_family_jpeg_qp80
