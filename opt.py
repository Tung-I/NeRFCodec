import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    ####
    group = parser.add_argument_group("jpeg_ste")
    group.add_argument("--codec_backend", type=str, default="jpeg", choices=["jpeg", "adaptor"])
    group.add_argument("--jpeg_quality", type=int, default=85)
    group.add_argument("--jpeg_plane_packing_mode", type=str, default="flatten")    # or "sandwich", etc.
    group.add_argument("--jpeg_quant_mode", type=str, default="global")          # we use "global"
    group.add_argument("--jpeg_global_min", type=float, default=-20.0)
    group.add_argument("--jpeg_global_max", type=float, default=20.0)
    group.add_argument("--jpeg_align", type=int, default=64)
    group.add_argument("--ste_enabled", type=int, default=1)  
    ####
    parser.add_argument('--resume_system_ckpt', action='store_true', default=False)
    parser.add_argument('--resume_optim', action='store_true', default=False)
    parser.add_argument('--extra_iters', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=5000)
    ### #
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument("--add_timestamp", type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument("--add_exp_version", type=int, default=0,
                        help='add experiment version ID to dir')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')

    parser.add_argument('--with_depth', action='store_true')
    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)

    parser.add_argument('--model_name', type=str, default='TensorVMSplit',
                        choices=['TensorVMSplit', 'TensorCP', 'TriPlane', 'TensorSTE'])

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000)

    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'nsvf', 'dtu','tankstemple', 'own_data', 'rtmv', 'miv', 'lvc', 'lvc_ff'])


    # training options
    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.02,
                        help='learning rate')    
    parser.add_argument("--lr_basis", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help = 'number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')

    # loss
    parser.add_argument("--L1_weight_inital", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0,
                        help='loss weight')
    parser.add_argument("--Ortho_weight", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_density", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=0.0,
                        help='loss weight')
    
    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25,
                        help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')
                        
    # network decoder
    parser.add_argument("--shadingMode", type=str, default="MLP_PE",
                        help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=6,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=6,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')
    


    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)

    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default='softplus')
    parser.add_argument('--ndc_ray', type=int, default=0)
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)


    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')



    parser.add_argument('--N_voxel_init',
                        type=int,
                        default=100**3)
    parser.add_argument('--N_voxel_final',
                        type=int,
                        default=300**3)
    parser.add_argument("--upsamp_list", type=int, action="append")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")

    parser.add_argument('--idx_view',
                        type=int,
                        default=0)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5,
                        help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=1000,
                        help='frequency of visualize the image')

    # compression
    parser.add_argument("--compression", action='store_true',
                        help='')
    parser.add_argument("--rate_penalty", action='store_true',
                        help='')
    parser.add_argument("--codec_training", action='store_true',
                        help='')
    parser.add_argument("--compression_strategy", type=str, default='batchwise_img_coding',
                        choices=['batchwise_img_coding', 'adaptor_feat_coding'])
    parser.add_argument("--codec_ckpt", type=str, default='')
    parser.add_argument("--compress_before_volrend", action='store_true',
                        help='')

    # compression experiment options
    parser.add_argument("--codec_backbone_type", type=str, default='cheng2020-anchor',
                        choices=['bmshj2018-hyperprior', 'cheng2020-anchor', 'mbt2018-mean'])
    parser.add_argument("--fix_triplane", type=int, default=0,
                        help='')
    parser.add_argument("--fix_encoder", action='store_true',
                        help='')
    parser.add_argument("--feat_rec_loss", type=int, default=0,
                        help='')
    parser.add_argument("--lr_feat_codec", type=float, default=2e-4,
                        help='')
    parser.add_argument("--lr_aux", type=float, default=1e-2,
                        help='')

    parser.add_argument("--lr_reset", type=int, default=0,
                        help='reset lr to inital')

    parser.add_argument("--den_rate_weight", type=float, default=1.0,
                        help='loss weight')
    parser.add_argument("--app_rate_weight", type=float, default=1.0,
                        help='loss weight')
    parser.add_argument("--warm_up_ckpt", type=str, default='')

    parser.add_argument("--entropy_on_weight", action='store_true',
                        help='')

    parser.add_argument("--warm_up", action='store_true',
                        help='')
    parser.add_argument("--warm_up_iters", type=int, default=10000,
                        help='')
    parser.add_argument("--decode_from_latent_code", action='store_true',
                        help='')
    parser.add_argument("--lr_latent_code", type=float, default=0.002,
                        help='')

    parser.add_argument("--adaptor_q_bit", type=int, default=8,)

    parser.add_argument("--joint_train_from_scratch", action='store_true',
                        help='')




    # finetune
    parser.add_argument("--fix_decoder_prior", action='store_true',
                        help='fix part of decoder in coding network(exclude adaptor)')
    parser.add_argument("--additional_vec", action='store_true',)

    parser.add_argument("--resume_finetune", action='store_true',)
    parser.add_argument("--system_ckpt", type=str, default=None)

    parser.add_argument("--vec_qat", action='store_true',
                        help='')

    # shared mlp
    parser.add_argument("--shared_mlp", type=int, default=0,
                        help='if true, do not update mlp, only update triplane')

    # dynamic scene
    parser.add_argument("--frame_idx", type=int, default=0,
                        help='')
    parser.add_argument("--start_frame_idx", type=int, default=0,
                        help='')
    parser.add_argument("--end_frame_idx", type=int, default=0,
                        help='')
    parser.add_argument("--temporal_consistent_weight", type=float, default=1e-2,
                        help='')


    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()

def codec_trainer_config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    # parser.add_argument('--config', is_config_file=True,
    #                     help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log/feat_codec',
                        help='where to store ckpts and logs')
    parser.add_argument("--add_timestamp", type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument("--add_exp_version", type=int, default=0,
                        help='add experiment version ID to dir')

    ### param. to be optimized
    parser.add_argument("--fix_TensoRF", action='store_true')

    ### training loop setting
    parser.add_argument("--repeat_times_each_scene", type=int, default=100)
    parser.add_argument("--iters_per_scene", type=int, default=1)
    parser.add_argument("--eval_rounds", type=int, default=10,
                        help='eval interval')
    parser.add_argument("--batch_size", type=int, default=4096)


    ### load codec ckpt setting
    parser.add_argument("--codec_ckpt", type=str, default=None)

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()