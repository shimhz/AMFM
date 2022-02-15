from comet_ml import Experiment
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import os
import struct
import torch
import argparse
import json
import sys
import torch
import importlib
import soundfile as sf
import pickle as pk
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from m_parser import get_args
from m_dataloaders import get_loaders
from m_trainer import *
from utils import *


def main():
    # parse arguments
    args = get_args()

    # load comet
    experiment = Experiment(
        api_key="UF4lkYcpEv0dfJqjTIj7NzKwd",
        project_name="dcase_ranked",
        workspace="shimhz",
        auto_output_logging=False,
        auto_metric_logging=False,
        disabled=args.comet_disable,
    )
    experiment.set_name(args.name)

    # device setting
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # get DB list
    lines = get_utt_list(args.DB_dir + args.wav_dir)
    print("nb_utt: {}".format(len(lines)))
    experiment.log_parameter("nb_utt", len(lines))

    # get label dictionary
    loader_args = {}
    if args.make_d_label:
        with open(args.DB_dir + args.meta_scp) as f:
            l_meta = f.readlines()
        (
            loader_args["d_label"],
            loader_args["l_label"],
            loader_args["d_abs_label"],
            loader_args["l_abs_label"],
        ) = make_d_label(l_meta[1:])
        pk.dump(
            [loader_args["d_label"], loader_args["l_label"]],
            open(args.DB_dir + args.d_label_dir, "wb"),
        )
        pk.dump(
            [loader_args["d_abs_label"], loader_args["l_abs_label"]],
            open(args.DB_dir + args.d_abs_label_dir, "wb"),
        )
    else:
        loader_args["d_label"], loader_args["l_label"] = pk.load(
            open(args.DB_dir + args.d_label_dir, "rb")
        )
        loader_args["d_abs_label"], loader_args["l_abs_label"] = pk.load(
            open(args.DB_dir + args.d_abs_label_dir, "rb")
        )
    print(loader_args["d_label"])
    print(loader_args["d_abs_label"])

    # split trnset and devset
    loader_args["trn_lines"] = split_dcase2020_fold_strict(
        fold_scp=args.DB_dir + args.fold_trn, lines=lines
    )
    loader_args["dev_lines"] = split_dcase2020_fold_strict(
        fold_scp=args.DB_dir + args.fold_scp, lines=lines
    )

    print(
        "#train utt: {}\t#dev utt: {}".format(
            len(loader_args["trn_lines"]), len(loader_args["dev_lines"])
        )
    )
    experiment.log_parameters(
        {"nb_trn_utt": len(loader_args["trn_lines"]), "nb_dev_utt": len(loader_args["dev_lines"])}
    )
    del lines

    if args.debug:
        np.random.shuffle(loader_args["trn_lines"])
        np.random.shuffle(loader_args["dev_lines"])
        loader_args["trn_lines"] = loader_args["trn_lines"][:1000]
        loader_args["dev_lines"] = loader_args["dev_lines"][:1000]

    # define dataset generators
    trnset_gen, devset_gen = get_loaders(loader_args, args)

    # set save directory
    save_dir = args.save_dir + args.name + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + "results/"):
        os.makedirs(save_dir + "results/")
    if not os.path.exists(save_dir + "weights/"):
        os.makedirs(save_dir + "weights/")

    # log experiment parameters to local and comet_ml server
    f_params = open(save_dir + "f_params.txt", "w")
    for k, v in sorted(vars(args).items()):
        print(k, v)
        f_params.write("{}:\t{}\n".format(k, v))
    f_params.close()
    experiment.log_parameters(vars(args))

    # define model
    module = importlib.import_module(args.model_scp)
    _model = getattr(module, args.model_name)
    model = _model(**args.model).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params: %d" % nb_params)
    experiment.log_parameter("nb_params", nb_params)

    # set ojbective funtions
    criterion = nn.CrossEntropyLoss()

    # set optimizer
    params = list(model.parameters())
    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.opt_mom, weight_decay=args.wd, nesterov=args.nesterov
        )
    elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad
        )
    else:
        raise NotImplementedError("Optimizer not implemented, got:{}".format(args.optimizer))

    # set learning rate decay
    if bool(args.do_lr_decay):
        if args.lr_decay == "keras":
            # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: keras_lr_decay(step))
            raise NotImplementedError("Not implemented yet")
        elif args.lr_decay == "cosine":
            lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=len(trnset_gen) * args.lrdec_t0, eta_min=0.000001
            )
        elif args.lr_decay == "triangle2":
            args.nb_iter_per_epoch = int(len(trnset_gen) - (len(trnset_gen) % 100))
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=args.lr_min,
                step_size_up=args.nb_iter_per_epoch * (args.epoch_per_cycle // 2),
                max_lr=args.lr,
                mode="triangular2",
                cycle_momentum=False,
            )
        else:
            raise NotImplementedError("Not implemented yet")
    else:
        lr_scheduler = None

    best_acc = 0.0
    best_abs_acc = 0.0
    f_acc = open(save_dir + "accs.txt", "a", buffering=1)
    for epoch in tqdm(range(args.epoch)):

        train(
            model=model,
            trnset_gen=trnset_gen,
            epoch=epoch,
            experiment=experiment,
            args=args,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        best_acc, acc, best_abs_acc, abs_acc = validate_TTA(
            model=model,
            devset_gen=devset_gen,
            epoch=epoch,
            device=device,
            l_label=loader_args["l_label"],
            best_acc=best_acc,
            best_abs_acc=best_abs_acc,
            args=args,
            save_dir=save_dir,
            experiment=experiment,
        )
        f_acc.write("%d %f %f\n" % (epoch, float(acc), float(abs_acc)))
    f_acc.close()


if __name__ == "__main__":
    main()