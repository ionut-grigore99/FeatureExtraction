import os
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import lovely_tensors as lt
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm

from ..datasets.coco import Coco
from ..datasets.kitti import Kitti
from .keypoints_detector import KeypointDetector


class Trainer:
    def __init__(self, model, optim, train_dataloader, eval_dataloader, *, epochs, writer_config):
        self.model = model
        self.optim = optim
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.epochs = epochs
        self.criterion = partial(sigmoid_focal_loss, reduction="mean")
        self.writer = SummaryWriter(**writer_config)
        self.scheduler = None
        self._train_step = 0
        self._eval_step  = 0
        self._epoch_step = 0
        self.architecture = self.model.print_model()
        self.writer.add_text("model/architecture", self.architecture.__repr__())

        torch.backends.cudnn.enabled = False
        # if torch.cuda.is_available():
        #     self.model.cuda()
        #     if torch.cuda.device_count() > 1:
        #         self.model = torch.nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        lt.monkey_patch()

    def fit(self):
        self._train_step = 0
        self._eval_step = 0
        for epoch in (tbar := tqdm(range(self.epochs), desc="Epoch")):
            train_loss = self.train_epoch()
            eval_loss = self.eval_epoch()
            tbar.set_postfix({"Train Loss": train_loss, "Eval Loss": eval_loss})
            self._draw_preds()
            self._save_checkpoint()
            self._epoch_step += 1

    def train_epoch(self):
        train_loss = 0
        self.model.train()
        for i, train_dataloader in enumerate(self.train_dataloader):
            for batch in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Train/{i}"):
                train_loss += self.train_batch(batch)
        train_loss /= self.train_samples
        return train_loss

    @torch.no_grad()
    def eval_epoch(self):
        epoch_loss = 0
        self.model.eval()
        for i, eval_dataloader in enumerate(self.eval_dataloader):
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc=f"Eval/{i}"):
                epoch_loss += self.eval_batch(batch)
        epoch_loss /= self.eval_samples
        return epoch_loss

    def train_batch(self, batch):
        inputs, targets = self._process_batch(batch)
        assert targets.min() == 0. and targets.max() == 1.
        self.optim.zero_grad()
        heatmaps = self.model(inputs)
        loss = self.criterion(heatmaps.squeeze(), targets.squeeze())
        loss.backward()
        self.optim.step()
        if self.scheduler is not None: self.scheduler.step()
        self.writer.add_scalar("train/loss", loss, self._train_step)
        self._train_step += 1
        return loss.item()

    def eval_batch(self, batch):
        inputs, targets = self._process_batch(batch)
        assert targets.min() == 0. and targets.max() == 1.
        heatmaps = self.model(inputs)
        loss = self.criterion(heatmaps.squeeze(), targets.squeeze())
        self.writer.add_scalar("eval/loss", loss, self._eval_step)
        if self._eval_step == 0: self._batch_reserved_for_writer = batch
        self._eval_step += 1
        return loss.item()

    def overfit_batch(self, iters=100):
        batch = next(iter(self.train_dataloader[0]))
        self._batch_reserved_for_writer = batch
        inputs, targets = self._process_batch(batch)
        for i in (tbar := tqdm(range(iters), desc="Overfit")):
            self.optim.zero_grad()
            heatmaps = self.model(inputs)
            breakpoint()
            loss = self.criterion(heatmaps.squeeze(), targets.squeeze())
            loss.backward()
            self.optim.step()
            tbar.set_postfix({"Loss": loss.item()})
            self.writer.add_scalar("overfit/loss", loss.item(), i)
            self._draw_preds()
            self._eval_step += 1
        return loss.item()

    def register_scheduler(self, scheduler):
        self.scheduler = scheduler

    @property
    def train_samples(self): return sum([len(_) for _ in self.train_dataloader])

    @property
    def eval_samples(self): return sum([len(_) for _ in self.eval_dataloader])

    # private methods
    # ---------------
    def _process_batch(self, batch):
        inputs, targets = batch
        # if torch.cuda.is_available():
        #     inputs = inputs.cuda()
        #     targets = targets.cuda()
        return inputs, targets

    @torch.no_grad()
    def _draw_preds(self, num_samples=5):
        inputs, targets = self._process_batch(self._batch_reserved_for_writer)
        num_samples = min(num_samples, inputs.shape[0])
        self.model.eval()
        heatmaps = self.model(inputs)
        for n in range(inputs.shape[0]):
            frames  = inputs[n].cpu().numpy().squeeze()
            heatmap = heatmaps[n].cpu().numpy().squeeze()
            target  = targets[n].cpu().numpy().squeeze()
            j, i = np.where(target > 0.5) # because the mask is label smoothed
            plt.style.use("classic")
            fig = plt.figure(figsize=(24, 12))
            plt.imshow(frames, cmap="gray")
            plt.imshow(gaussian_filter(heatmap, 5), cmap="magma", alpha=.5)
            plt.scatter(i, j, color="yellow", s=50, marker="+")
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.axis("off")
            self.writer.add_figure(f"pred/{n}", fig, self._eval_step)
            plt.close(fig)
            if n == num_samples: break

    def _save_checkpoint(self):
        with open(Path(self.writer.logdir) / f"ckp-epoch={self._epoch_step}-step={self._train_step}.pth", "wb") as f:
            model_state = {
                'model_state_dict': deepcopy(self.model.state_dict()),
                'optimizer_state_dict': deepcopy(self.optim.state_dict())
            }
            torch.save(model_state, f)


def find_folder(root_dir, target_folder):
    found_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_folder in dirnames:
            full_path = Path(dirpath) / target_folder / "data"
            found_paths.append(full_path.as_posix())
    return found_paths


def get_train_val(dset, bs):
    if dset == "coco":
        train_dataloader = [
            DataLoader(
                Coco(
                    root_dir="/data/COCO/train2017"),
                    shuffle=True,
                    batch_size=bs,
                    pin_memory=True
            )
        ]
        eval_dataloader = [
            DataLoader(
                Coco(
                    root_dir="/data/COCO/val2017"),
                    shuffle=True,
                    batch_size=bs,
                    pin_memory=True
            )
        ]
    elif dset == "kitti":
        kitti_root_dir = "/data/disertatie/kitti"
        kitti = find_folder(kitti_root_dir, "image_00")
        train_dataloader = [
            DataLoader(
                Kitti(
                    root_dir=recording),
                    shuffle=False,  # @NOTE: shuffle is False because we want to keep the temporal order of the frames
                    batch_size=bs,
                    pin_memory=True
            ) for recording in kitti
        ]
        eval_dataloader = [
            DataLoader(
                Kitti(
                    root_dir=recording),
                    shuffle=False,
                    batch_size=bs,
                    pin_memory=True
            ) for recording in kitti
        ]
    return train_dataloader, eval_dataloader


if __name__ == "__main__":
    epochs = 10
    lr = 1e-3
    weight_decay = 1e-4
    annealing_factor = 0.01
    writer_config = {
        "logdir": Path("/data/SP_FPN/runs") / datetime.now().strftime("%Y%m%d-%H%M%S"),
        "flush_secs": 10,
        "write_to_disk": True,
    }
    # bootstrap
    train_dataloaders, eval_dataloaders = get_train_val("coco", 8)
    model = KeypointDetector()
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainer = Trainer(model, optim, train_dataloaders, eval_dataloaders, epochs=epochs, writer_config=writer_config)
    scheduler = CosineAnnealingLR(optim, T_max=trainer.train_samples * epochs, eta_min=annealing_factor * lr)
    trainer.register_scheduler(scheduler)
    trainer.overfit_batch(1000)
    #trainer.fit()

