import argparse
import copy
import csv
import gc
import os
import random
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import utils
from dataset import CustomDataset
from load_datasets import load_datasets
from losses import CustomLoss
from models import create_model
from schedulefree import RAdamScheduleFree
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer(object):
    def __init__(self, args):
        self.config = utils.load_config(args.config_path)

        self._set_seed()

        self.device = utils.set_device(args.gpu_id)
        self.device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.use_cuda = self.device.type == "cuda"

        self.use_fp16 = args.fp16
        if self.use_fp16:
            self.scaler = torch.amp.GradScaler()

        self.accumulate = max(round(64 / self.config["training"]["batch_size"]), 1)

        self.save_dir = os.path.join(self.config["training"]["save_dir"], "train")
        os.makedirs(self.save_dir, exist_ok=True)

        self._init_model()
        self._init_data()
        self._init_loss()
        self._init_optimizer()

        self.logger, self.log_file = self._setup_logger(os.path.join(self.save_dir, "log.csv"))

        self.avg_iou_loss = 0.0
        self.avg_conf_loss = 0.0
        self.avg_cls_loss = 0.0
        self.train_loss = 0.0
        self.best_train_loss = float("inf")
        self.val_loss = 0.0
        self.best_val_loss = float("inf")
        self.start_epoch = 1
        self.global_step = 0

    def _set_seed(self):
        random.seed(self.config["training"]["seed"])
        np.random.seed(self.config["training"]["seed"])
        torch.manual_seed(self.config["training"]["seed"])

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config["training"]["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def _init_model(self):
        self.model = create_model(self.config["model"]["type"], self.config["model"]["num_classes"])
        self.model.to(self.device)

        if self.config["training"]["use_ema"]:
            self.ema = utils.EMA(self.model)

    def _init_data(self):
        train_data, train_labels, val_data, val_labels = load_datasets(self.config["datasets"])
        train_dataset = CustomDataset(train_data, train_labels, self.config["datasets"], training=True)

        self.has_validation_data = len(val_data) > 0

        if self.has_validation_data:
            val_dataset = CustomDataset(val_data, val_labels, self.config["datasets"], training=False)
        else:
            val_dataset = CustomDataset([], [], self.config["datasets"], training=False)

        if self.config["datasets"]["num_workers"] > 0:
            persistent_workers = True
        else:
            persistent_workers = False

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["datasets"]["num_workers"],
            pin_memory=self.use_cuda,
            persistent_workers=persistent_workers,
            collate_fn=CustomDataset.custom_collate_fn,
        )

        if self.has_validation_data:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                num_workers=self.config["datasets"]["num_workers"],
                pin_memory=self.use_cuda,
                persistent_workers=persistent_workers,
                collate_fn=CustomDataset.custom_collate_fn,
            )
        else:
            self.val_loader = None

    def _init_loss(self):
        self.loss_func = CustomLoss(self.config["model"]["num_classes"], self.device, self.use_fp16)

    def _init_optimizer(self):
        optimizer_type = self.config["training"]["optimizer"]
        if optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                utils.set_params(self.model, self.config["training"]["weight_decay"]),
                self.config["training"]["learning_rate"],
                self.config["training"]["momentum"],
                nesterov=self.config["training"].get("nesterov", True),
            )
        elif optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config["training"]["learning_rate"],
                betas=(self.config["training"]["beta1"], self.config["training"]["beta2"]),
            )
        elif optimizer_type == "radam_schedulefree":
            self.optimizer = RAdamScheduleFree(
                self.model.parameters(),
                lr=self.config["training"]["learning_rate"],
                betas=(self.config["training"]["beta1"], self.config["training"]["beta2"]),
                eps=self.config["training"]["eps"],
                weight_decay=self.config["training"]["weight_decay"],
            )
        else:
            print(f"Unsupported optimizer type: {optimizer_type}. Choose 'sgd', 'adam', or 'radam_schedulefree'.")
            sys.exit(1)

        if self.config["training"]["use_scheduler"] and optimizer_type != "radam_schedulefree":
            scheduler_type = self.config["training"]["scheduler"]
            if scheduler_type == "cosine":
                self.scheduler = utils.CosineLR(
                    self.config["training"]["learning_rate"],
                    self.config["training"]["max_lr"],
                    self.config["training"]["warmup_epochs"],
                    self.config["training"]["total_epochs"],
                    len(self.train_loader),
                )
            elif scheduler_type == "linear":
                self.scheduler = utils.LinearLR(
                    self.config["training"]["learning_rate"],
                    self.config["training"]["max_lr"],
                    self.config["training"]["warmup_epochs"],
                    self.config["training"]["total_epochs"],
                    len(self.train_loader),
                )
            else:
                print(f"Unsupported scheduler type: {scheduler_type}. Choose 'cosine' or 'linear'.")
                sys.exit(1)

    def _setup_logger(self, log_path):
        fieldnames = ["epoch", "iou_loss", "conf_loss", "cls_loss", "train_total_loss", "val_total_loss"]

        log_file = open(log_path, "w")
        logger = csv.DictWriter(log_file, fieldnames=fieldnames)
        logger.writeheader()

        return logger, log_file

    def _train_epoch(self, epoch):
        self.model.train()

        if self.config["training"]["optimizer"] == "radam_schedulefree":
            self.optimizer.train()

        self.optimizer.zero_grad()

        if epoch == (self.config["training"]["total_epochs"] + 1) - self.config["training"]["mosaic_off_epoch"]:
            self.train_loader.dataset.config["mosaic"] = 0.0

        self.avg_iou_loss = 0.0
        self.avg_conf_loss = 0.0
        self.avg_cls_loss = 0.0

        use_scheduler = self.config["training"]["use_scheduler"]

        with tqdm(self.train_loader, leave=False) as pbar:
            for iteration, (inputs, targets) in enumerate(pbar, 1):
                if use_scheduler and hasattr(self, "scheduler"):
                    self.scheduler.step(self.global_step, self.optimizer)

                inputs = inputs.to(self.device)
                targets = [target.to(self.device) for target in targets]

                with torch.amp.autocast(device_type=self.device_type, enabled=self.use_fp16):
                    outputs = self.model(inputs)
                    outputs = [torch.nan_to_num(output) for output in outputs]
                    iou_loss, conf_loss, cls_loss = self.loss_func(outputs, targets)
                    total_loss = iou_loss + conf_loss + cls_loss

                self.avg_iou_loss += iou_loss.item()
                self.avg_conf_loss += conf_loss.item()
                self.avg_cls_loss += cls_loss.item()

                if self.use_fp16:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

                if self.global_step % self.accumulate == 0:
                    if self.use_fp16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                    if self.config["training"]["use_ema"] and hasattr(self, "ema"):
                        self.ema.update(self.model)

                self.memory = f"{torch.cuda.memory_reserved() / 1E9:.3f}G" if self.device.type == "cuda" else "N/A"
                current_str = ("%15s" * 2 + "%15.3f" * 3) % (
                    f"{epoch}/{self.config['training']['total_epochs']}",
                    self.memory,
                    iou_loss.item(),
                    conf_loss.item(),
                    cls_loss.item(),
                )

                pbar.set_description(current_str)
                pbar.set_postfix(loss=f"{total_loss.item():.3f}")

                self.global_step += 1
                time.sleep(0.1)

                del outputs, iou_loss, conf_loss, cls_loss, total_loss

        self.avg_iou_loss /= iteration
        self.avg_conf_loss /= iteration
        self.avg_cls_loss /= iteration
        self.train_loss = self.avg_iou_loss + self.avg_conf_loss + self.avg_cls_loss

    def _validate(self):
        if self.config["training"]["use_ema"] and hasattr(self, "ema"):
            eval_model = self.ema.ema
        else:
            eval_model = self.model

        eval_model.eval()

        if self.config["training"]["optimizer"] == "radam_schedulefree":
            self.optimizer.eval()

        self.val_loss = 0.0

        with torch.no_grad():
            with tqdm(self.val_loader, leave=False) as pbar:
                for iteration, (inputs, targets) in enumerate(pbar, 1):
                    inputs = inputs.to(self.device)
                    targets = [target.to(self.device) for target in targets]

                    outputs = eval_model(inputs)
                    outputs = [torch.nan_to_num(output) for output in outputs]

                    iou_loss, conf_loss, cls_loss = self.loss_func(outputs, targets)
                    total_loss = iou_loss + conf_loss + cls_loss
                    self.val_loss += total_loss.item()

                    pbar.set_description("[Validating]")
                    pbar.set_postfix(loss=total_loss.item())
                    time.sleep(0.1)

        self.val_loss /= iteration

    def _save_checkpoints(self, epoch):
        if self.config["training"]["use_ema"] and hasattr(self, "ema"):
            save_model = copy.deepcopy(self.ema.ema)
        else:
            save_model = copy.deepcopy(self.model)

        checkpoint = {
            "epoch": epoch,
            "model": save_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        if self.config["training"]["optimizer"] == "radam_schedulefree":
            self.optimizer.eval()

        torch.save(checkpoint, os.path.join(self.save_dir, "last.pt"))

        if self.has_validation_data:
            if self.val_loss <= self.best_val_loss:
                torch.save(checkpoint, os.path.join(self.save_dir, "best.pt"))
                self.best_val_loss = self.val_loss
        else:
            if self.train_loss <= self.best_train_loss:
                torch.save(checkpoint, os.path.join(self.save_dir, "best.pt"))
                self.best_train_loss = self.train_loss

    def train(self):
        print(("\n" + "%15s" * 5) % ("epoch", "memory", "iou_loss", "conf_loss", "cls_loss"))
        for epoch in range(self.start_epoch, self.config["training"]["total_epochs"] + 1):
            self._train_epoch(epoch)

            epoch_str = ("%15s" * 2 + "%15.3f" * 3) % (
                f"{epoch}/{self.config['training']['total_epochs']}",
                self.memory,
                self.avg_iou_loss,
                self.avg_conf_loss,
                self.avg_cls_loss,
            )
            print(epoch_str)

            log_data = {
                "epoch": f"{str(epoch).zfill(3)}",
                "iou_loss": str(f"{self.avg_iou_loss:.3f}"),
                "conf_loss": str(f"{self.avg_conf_loss:.3f}"),
                "cls_loss": str(f"{self.avg_cls_loss:.3f}"),
                "train_total_loss": str(f"{self.train_loss:.3f}"),
            }

            if self.has_validation_data:
                self._validate()
                log_data["val_total_loss"] = str(f"{self.val_loss:.3f}")
            else:
                log_data["val_total_loss"] = "N/A"

            self.logger.writerow(log_data)
            self.log_file.flush()

            self._save_checkpoints(epoch)
            torch.cuda.empty_cache()

        gc.collect()

        self.log_file.close()


def main(args):
    trainer = Trainer(args)
    trainer.train()

    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training program for object-detection model")
    parser.add_argument(
        "--config_path", type=str, default="config/config.yml", help="Path to the YAML configuration file"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=0, help="Random seeds")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training (FP16)")
    args = parser.parse_args()

    main(args)
