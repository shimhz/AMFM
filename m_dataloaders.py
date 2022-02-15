import torch
import numpy as np
import soundfile as sf
import torchaudio as ta

from torch.utils import data


def get_loaders(loader_args, args):
    trnset = Dataset_DCASE2020_t1(
        lines=loader_args["trn_lines"],
        nb_frames=args.nb_frames,
        base_dir=args.DB_dir + args.wav_dir,
        d_label=loader_args["d_label"],
        d_abs_label=loader_args["d_abs_label"],
    )
    trnset_gen = data.DataLoader(
        trnset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.nb_worker,
        pin_memory=True,
        drop_last=True,
    )

    devset = Dataset_DCASE2020_t1(
        lines=loader_args["dev_lines"],
        nb_frames=args.nb_frames,
        mode_trn=False,
        base_dir=args.DB_dir + args.wav_dir,
        d_label=loader_args["d_label"],
        d_abs_label=loader_args["d_abs_label"],
    )
    devset_gen = data.DataLoader(
        devset,
        batch_size=args.bs // 3,
        shuffle=False,
        num_workers=args.nb_worker // 2,
        pin_memory=True,
        drop_last=False,
    )

    return trnset_gen, devset_gen


class Dataset_DCASE2020_t1(data.Dataset):
    def __init__(
        self,
        lines,
        nb_frames=0,
        mode_trn=True,
        base_dir="",
        d_label="",
        d_abs_label="",
        return_label=True,
        samp_rate=44100,
        pre_emp=True,
        n_fft=2048,
        nb_mels=128,
        win_len=40,
        hop_len=20,
        do_mvn=True,
    ):
        self.lines = lines
        self.d_label = d_label
        self.d_abs_label = d_abs_label
        self.base_dir = base_dir
        self.nb_frames = nb_frames
        self.mode_trn = mode_trn
        self.pre_emp = pre_emp
        self.return_label = return_label
        self.do_mvn = do_mvn

        self.samp_rate = samp_rate
        self.melspec = ta.transforms.MelSpectrogram(
            samp_rate,
            n_fft=n_fft,
            win_length=int(samp_rate * 0.001 * win_len),
            hop_length=int(samp_rate * 0.001 * hop_len),
            window_fn=torch.hamming_window,
            n_mels=nb_mels,
        )
        # self.tot_frame = int(441000 / (44100 * 0.001 * hop_len))
        self.nb_samps = int(samp_rate * 0.001 * hop_len * nb_frames)
        self.margin = int(441000 - self.nb_samps)
        if not mode_trn:
            self.TTA_mid_idx = int(self.nb_samps / 2)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        k = self.lines[index]
        try:
            # X, samp_rate = ta.load(self.base_dir + k, normalization=True)  # X.size : (1, 441000)
            X, samp_rate = ta.load(self.base_dir + k)

        except:
            raise ValueError("Unable to laod utt %s" % k)
        if self.pre_emp:
            X = self._pre_emphasis(X)

        if self.mode_trn:
            st_idx = np.random.randint(0, self.margin)
            X = X[:, st_idx : st_idx + self.nb_samps]
        else:
            l_X = []
            l_X.append(X[:, : self.nb_samps])
            l_X.append(X[:, self.TTA_mid_idx : self.TTA_mid_idx + self.nb_samps])
            l_X.append(X[:, -self.nb_samps :])
            X = torch.stack(l_X)
        # print (X.size())
        # X = self.melspec(X)
        # print (X.size())
        # if self.do_mvn: X = self._utt_mvn(X)
        # print(X.size())    #trn: (1, 128, 251) dev: (3, 128, 251)
        if self.return_label:
            y = self.d_label[k.split("-")[0]]  # scene
            y_abs = self.d_abs_label[k.split("-")[0]]
            # k = k.split('-')[-1] # device
            # y = self.d_label[k.split('.')[0]] # device
            return X, y, y_abs
        else:
            return X

    def _pre_emphasis(self, x):
        return x[:, 1:] - 0.97 * x[:, :-1]

    def _utt_mvn(self, x):
        _m = x.mean(dim=-1, keepdim=True)
        _s = x.std(dim=-1, keepdim=True)
        _s[_s < 0.001] = 0.001
        return (x - _m) / _s
