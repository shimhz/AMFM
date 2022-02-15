import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix

from utils import *
from augment.specaugment import spec_augment
from augment.mixup import mixup_data_joint, mixup_criterion


def train(model, trnset_gen, epoch, experiment, args, device, criterion, optimizer, lr_scheduler):
    # train phase
    model.train()
    idx_ct_start = int(len(trnset_gen) * epoch)

    loss_cce = 0.0
    loss_abs_cce = 0.0
    ratio = args.ratio

    # mixup = True if epoch > args.mixup_start else False
    with tqdm(total=len(trnset_gen), ncols=70) as pbar:
        for idx, (m_batch, m_label, m_abs_label) in enumerate(trnset_gen):

            if args.warm:
                warmup_learning_rate(args, epoch, idx, len(trnset_gen), optimizer)

            m_batch, m_label, m_abs_label = (
                m_batch.to(device),
                m_label.to(device),
                m_abs_label.to(device),
            )
            m_batch = extract_melspec(m_batch, device)
            m_batch = m_batch.to(device)

            m_batch, m_label_1a, m_label_1b, m_abs_label_1a, m_abs_label_1b, lam = mixup_data_joint(
                m_batch, m_label, m_abs_label, alpha=args.mixup_alpha, use_cuda=True
            )
            m_batch, m_label_1a, m_label_1b, m_abs_label_1a, m_abs_label_1b = map(
                torch.autograd.Variable,
                [m_batch, m_label_1a, m_label_1b, m_abs_label_1a, m_abs_label_1b],
            )

            spec_batch = torch.unsqueeze(spec_augment(mel_spectrogram=torch.squeeze(m_batch, 1)), 1)

            # mid_output, output = model(m_batch) # fixed mixup
            mid_output, output = model(spec_batch)

            _loss_abs_cce = mixup_criterion(
                criterion, mid_output, m_abs_label_1a, m_abs_label_1b, lam
            )
            _loss_cce = mixup_criterion(criterion, output, m_label_1a, m_label_1b, lam)

            loss = int(ratio) * _loss_cce + _loss_abs_cce
            loss_cce += _loss_cce.detach().cpu().numpy()
            loss_cce += _loss_abs_cce.detach().cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description("epoch: %d cce: %.3f" % (epoch, loss.detach().cpu().numpy()))
            pbar.update(1)
            if idx % args.nb_iter_per_log == 0:
                if idx != 0:
                    loss_cce /= args.nb_iter_per_log

                experiment.log_metric(
                    "loss_spec", _loss_cce.detach().cpu().numpy(), step=idx_ct_start + idx
                )
                experiment.log_metric(
                    "loss_abs", _loss_abs_cce.detach().cpu().numpy(), step=idx_ct_start + idx
                )
                experiment.log_metric("loss", loss_cce, step=idx_ct_start + idx)

                loss_cce = 0

                for p_group in optimizer.param_groups:
                    lr = p_group["lr"]
                    break
                experiment.log_metric("lr", lr, step=idx_ct_start + idx)

            if args.do_lr_decay:
                lr_scheduler.step()


def validate_TTA(
    model,
    devset_gen,
    epoch,
    experiment,
    device,
    l_label,
    best_acc,
    best_abs_acc,
    save_dir,
    args,
):
    # validation phase
    model.eval()
    with torch.set_grad_enabled(False):
        y_pred = []
        y_true = []

        abs_y_pred = []
        abs_y_true = []
        with tqdm(total=len(devset_gen), ncols=70) as pbar:
            for m_batch, m_label, m_abs_label in devset_gen:
                # print(m_batch.size())#(8,3,220500)
                m_batch = extract_melspec(m_batch, device, args.bs // 3)
                m_batch = m_batch.view(-1, 1, 256, args.nb_frames + 1).to(device)
                mid_out, out = model(m_batch)

                out = F.softmax(out, dim=-1).view(-1, 3, out.size(1)).mean(dim=1, keepdim=False)
                mid_out = (
                    F.softmax(mid_out, dim=-1)
                    .view(-1, 3, mid_out.size(1))
                    .mean(dim=1, keepdim=False)
                )

                m_label = list(m_label.numpy())
                m_abs_label = list(m_abs_label.numpy())

                y_pred.extend(list(out.cpu().numpy()))  # >>> (16, 64?)
                y_true.extend(m_label)

                abs_y_pred.extend(list(mid_out.cpu().numpy()))
                abs_y_true.extend(m_abs_label)

                pbar.set_description("epoch%d: Extract ValEmbeddings" % (epoch))
                pbar.update(1)

        y_pred = np.argmax(np.array(y_pred), axis=1).tolist()
        # print(y_pred == y_true)
        conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
        nb_cor = 0
        for i in range(len(conf_mat)):
            nb_cor += conf_mat[i, i]
            conf_mat[i, i] = 0
        acc = nb_cor / len(y_true) * 100

        abs_y_pred = np.argmax(np.array(abs_y_pred), axis=1).tolist()
        abs_conf_mat = confusion_matrix(y_true=abs_y_true, y_pred=abs_y_pred)
        abs_nb_cor = 0
        for i in range(len(abs_conf_mat)):
            abs_nb_cor += abs_conf_mat[i, i]
            abs_conf_mat[i, i] = 0
        abs_acc = abs_nb_cor / len(abs_y_true) * 100

        experiment.log_metric("acc1", acc, step=epoch)
        experiment.log_metric("abs_acc", abs_acc, step=epoch)

        # record best validation model
        if acc > best_acc:
            print("New best acc: %f" % float(acc))
            best_acc = acc
            experiment.log_metric("best_acc", best_acc, step=epoch)
            experiment.log_confusion_matrix(
                matrix=conf_mat, labels=l_label, step=epoch, overwrite=True, title="epoch%d" % epoch
            )

            torch.save(model.state_dict(), save_dir + "weights/best.pt")
        if not args.save_best_only:
            torch.save(model.state_dict(), save_dir + "weights/ep%d_%.4f.pt" % (epoch, acc))

        if abs_acc > best_abs_acc:
            print("New best abstract label acc: %f" % float(abs_acc))
            best_abs_acc = abs_acc
            experiment.log_metric("best_val_abs_acc", best_abs_acc, step=epoch)

    return best_acc, acc, best_abs_acc, abs_acc