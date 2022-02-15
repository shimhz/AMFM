import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # dir
    parser.add_argument("-name", type=str, required=True)
    parser.add_argument(
        "-DB_dir",
        type=str,
        default="/DB/TAU-urban-acoustic-scenes-2020-mobile-development/",
    )
    parser.add_argument("-meta_scp", type=str, default="meta.csv")
    parser.add_argument("-fold_scp", type=str, default="evaluation_setup/fold1_evaluate.csv")
    parser.add_argument("-fold_trn", type=str, default="evaluation_setup/fold1_train.csv")
    parser.add_argument("-save_dir", type=str, default="/result/dcase2020/")
    parser.add_argument("-wav_dir", type=str, default="audio/")
    parser.add_argument("-d_label_dir", type=str, default="d_label_scene.pk")
    parser.add_argument("-d_abs_label_dir", type=str, default="d_label_abstract.pk")

    # hyper-params
    parser.add_argument("-bs", type=int, default=24)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-lr_min", type=float, default=0.000005)
    parser.add_argument("-lr_decay", type=str, default="cosine")
    parser.add_argument("-lrdec_t0", type=int, default=80)
    parser.add_argument("-nb_frames", type=int, default=250)
    parser.add_argument("-wd", type=float, default=0.001)
    parser.add_argument("-mixup_start", type=int, default=5)
    parser.add_argument("-mixup_alpha", type=float, default=0.1)
    parser.add_argument("-epoch", type=int, default=800)
    parser.add_argument("-nb_iter_per_log", type=int, default=400)
    parser.add_argument("-nb_mels", type=int, default=256)
    parser.add_argument("-optimizer", type=str, default="sgd")
    parser.add_argument("-opt_mom", type=float, default=0.9)
    parser.add_argument("-nb_worker", type=int, default=12)
    parser.add_argument("-ratio", type=int, default=5)
    parser.add_argument("-epoch_per_cycle", type=int, default=40)

    parser.add_argument("-warm", type=str2bool, default=True)
    parser.add_argument("-warm_epochs", type=int, default=10)
    parser.add_argument("-warmup_from", type=float, default=0.001)
    parser.add_argument("-warmup_to", type=float, default=0.01)

    # DNN args
    # def __init__(self, block, nb_blocks = [3, 4, 6], nb_filts = [16, 32, 64], nb_strides = [1, 2, 2], nb_classes=10, reduction=16, nb_code = 64):
    parser.add_argument("-model_scp", type=str, required=True)
    parser.add_argument("-model_name", type=str, required=True)
    # parser.add_argument('-m_nb_blocks', type = list, default = [3, 4, 6])
    # parser.add_argument('-m_nb_filts', nargs='+', type = int, default = [16, 32, 64])
    # parser.add_argument('-m_nb_strides', type = list, default = [1, 2, 2])
    # parser.add_argument('-m_nb_strides', type = list, default = [1, (3,6), 5])
    # parser.add_argument('-m_reduction', type = int, default = 16)
    # parser.add_argument('-m_nb_code', type = int, default = 64)
    # parser.add_argument('-m_nb_classes', type = int, default = 10) # scene
    # parser.add_argument('-m_nb_classes', type = int, default = 9) # device

    # flag
    parser.add_argument("-amsgrad", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("-nesterov", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("-make_d_label", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("-debug", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("-comet_disable", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("-save_best_only", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("-do_lr_decay", type=str2bool, nargs="?", const=True, default=True)

    args = parser.parse_args()
    args.model = {}
    for k, v in vars(args).items():
        if k[:2] == "m_":
            print(k, v)
            args.model[k[2:]] = v
    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
