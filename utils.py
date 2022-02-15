import os
import torch
import numpy as np
import torchaudio as ta


def get_utt_list(src_dir):
    """
    Designed for DCASE2020 task 1-a
    """
    l_utt = []
    for r, ds, fs in os.walk(src_dir):
        for f in fs:
            if f[-3:] != "wav":
                continue
            k = f
            l_utt.append(k)

    return l_utt


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def make_d_label(lines, mode="scene"):
    idx = 0
    dic_label = {}
    list_label = []

    dic_abs_label = {}
    list_abs_label = ["indoor", "outdoor", "tranport"]

    l_indoor = ["airport", "shopping_mall", "metro_station"]
    l_outdoor = ["street_traffic", "street_pedestrian", "park", "public_square"]
    l_transport = ["bus", "metro", "tram"]

    for line in lines:

        if mode == "scene":
            label = line.strip().split("/")[1].split("-")[0]
        elif mode == "area":
            label = line.strip().split("/")[1].split("-")[1]
        elif mode == "device":
            tmp_label = line.strip().split("/")[1].split("-")[4]
            label = tmp_label.split(".")[0]

        if label in l_indoor:
            abs_label = "indoor"
            abs_idx = 0
        elif label in l_outdoor:
            abs_label = "outdoor"
            abs_idx = 1
        elif label in l_transport:
            if label != "metro_station":
                abs_label = "transport"
                abs_idx = 2

        if label not in dic_label:
            dic_label[label] = idx
            dic_abs_label[label] = abs_idx
            list_label.append(label)
            idx += 1

    return dic_label, list_label, dic_abs_label, list_abs_label


def split_dcase2020_fold(fold_scp, lines):
    """
    Input validate lines
    """
    fold_lines = open(fold_scp, "r").readlines()
    dev_lines = []
    val_lines = []

    fold_list = []
    for line in fold_lines[1:]:
        fold_list.append(line.strip().split("\t")[0].split("/")[1])
    for line in lines:
        if line in fold_list:
            val_lines.append(line)
        else:
            dev_lines.append(line)

    return dev_lines, val_lines


def split_dcase2020_fold_strict(fold_scp, lines):
    fold_lines = open(fold_scp, "r").readlines()
    l_return = []
    l_fold = []

    for line in fold_lines[1:]:
        l_fold.append(line.strip().split("\t")[0].split("/")[1])
    for line in lines:
        if line in l_fold:
            l_return.append(line)

    return l_return


def extract_melspec(
    m_batch,
    device,
    mode="trn",
    bs=24,
    samp_rate=44100,
    pre_emp=True,
    n_fft=2048,
    nb_mels=256,
    win_len=40,
    hop_len=20,
    do_mvn=True,
):

    X_batch = []
    # print (m_batch.size())
    # print (m_batch[0].size())
    melspec = ta.transforms.MelSpectrogram(
        samp_rate,
        n_fft=n_fft,
        win_length=int(samp_rate * 0.001 * win_len),
        hop_length=int(samp_rate * 0.001 * hop_len),
        window_fn=torch.hamming_window,
        n_mels=nb_mels,
    ).to(device)

    for i in range(bs):
        try:
            # print ('a' + str(m_batch.size()))
            X = melspec(m_batch[i].to(device))
            if do_mvn:
                X = utt_mvn(X)
            X_batch.append(torch.unsqueeze(X, 0))
        except:
            # print ('b' + str(m_batch.size()))
            last_batch = int(m_batch.shape[0])
            if i < last_batch:
                X = melspec(m_batch[i])
                if do_mvn:
                    X = utt_mvn(X)
                X_batch.append(torch.unsqueeze(X, 0))
            else:
                break

    return torch.cat(X_batch)


def utt_mvn(x):
    _m = x.mean(dim=-1, keepdim=True)
    _s = x.std(dim=-1, keepdim=True)
    _s[_s < 0.001] = 0.001

    return (x - _m) / _s


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))