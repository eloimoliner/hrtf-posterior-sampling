"""Training loop of the score model (HRTF prior)."""
import copy
import logging
import re
from glob import glob

import hydra
import numpy as np
import torch

logger = logging.getLogger(__name__)


class Trainer():

    def __init__(self, args, dset, network, diff_params, device):
        self.args = args
        self.dset = dset
        self.network = network
        self.diff_params = diff_params
        self.device = device

        self.optimizer = hydra.utils.instantiate(args.exp.optimizer, params=network.parameters())
        self.ema = copy.deepcopy(self.network).eval().requires_grad_(False)

        torch.manual_seed(self.args.exp.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False

        n_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        logger.info(f'total_params: {n_params / 1e6} M')

        self.it = 0
        if self.args.exp.resume:
            self.resume_from_checkpoint()
        self.losses = []

    def resume_from_checkpoint(self):
        """Resumes from the latest <exp_name>-<it>.pt in model_dir, if any."""
        id_regex = re.compile(rf"{self.args.exp.exp_name}-(\d*)\.pt")
        ids = [int(id_regex.search(p).groups()[0]) for p in glob(f"{self.args.model_dir}/{self.args.exp.exp_name}-*.pt")]
        if not ids:
            logger.info('no checkpoint found, training from scratch')
            return
        checkpoint = torch.load(f"{self.args.model_dir}/{self.args.exp.exp_name}-{max(ids)}.pt", map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['network'])
        self.ema.load_state_dict(checkpoint['ema'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.it = checkpoint['it']
        logger.info(f'resuming from iteration {self.it}')

    def state_dict(self):
        return {'it': self.it,
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'ema': self.ema.state_dict(),
                'args': self.args}

    def save_checkpoint(self):
        save_name = f"{self.args.model_dir}/{self.args.exp.exp_name}-{self.it}.pt"
        torch.save(self.state_dict(), save_name)
        logger.info(f'saved {save_name}')

    def get_batch(self):
        sample = next(self.dset)
        return {key: value.to(self.device) for key, value in sample.items()}

    def train_step(self):
        self.optimizer.zero_grad()
        sample = self.get_batch()
        error, sigma = self.diff_params.loss_fn(self.network, sample)
        loss = error.mean()
        loss.backward()

        if self.args.exp.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.exp.max_grad_norm)

        self.optimizer.step()
        self.losses.append(loss.item())

    def update_ema(self):
        """EMA of the network weights, with a linear ramp-up (in samples seen, it * exp.batch_size)."""
        ema_rampup = self.args.exp.ema_rampup
        ema_rate = self.args.exp.ema_rate
        t = self.it * self.args.exp.batch_size
        with torch.no_grad():
            if t < ema_rampup:
                s = np.clip(t / ema_rampup, 0.0, ema_rate)
                for dst, src in zip(self.ema.parameters(), self.network.parameters()):
                    dst.copy_(dst * s + src * (1 - s))
            else:
                for dst, src in zip(self.ema.parameters(), self.network.parameters()):
                    dst.copy_(dst * ema_rate + src * (1 - ema_rate))

    def training_loop(self):
        while self.it <= self.args.exp.max_iters:
            self.train_step()
            self.update_ema()

            if self.it > 0 and self.it % self.args.logging.save_interval == 0:
                self.save_checkpoint()

            if self.it > 0 and self.it % self.args.logging.log_interval == 0:
                logger.info(f'it {self.it}: loss {np.mean(self.losses):.4f}')
                self.losses = []

            self.it += 1
