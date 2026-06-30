import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.dataloader import DatasetSegmentation, RandomGenerator, ValGenerator
from trainers import *
import os
import argparse
import random
import numpy as np
from torch.nn.modules.loss import BCEWithLogitsLoss
import logging
from utils.main_utils import load_cfg_from_cfg_file, read_text
import torch.nn.functional as F


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str, help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Whether to resume training")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--data_percentage", type=int, default=100, help="Percentage of data to use.")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--teacher-ckpt-root", type=str, default="checkpoints", help="Root folder of downloaded teacher checkpoints.")
    parser.add_argument("--teacher-source", type=str, default="", help="Source dataset name for teacher checkpoint: BTMRI, BUSI, ISIC, Kvasir. Empty means infer from cfg.DATASET.NAME.")
    parser.add_argument("--no-init-student-from-teacher", action="store_true", help="Do not initialize student from teacher checkpoint.")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update({k: v for k, v in vars(args).items()})
    return cfg


def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding="UTF-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def calc_loss(low_res_logits, low_res_label_batch, ce_loss, dice_loss, cfg):
    loss_ce = ce_loss(low_res_logits, low_res_label_batch.float())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch)
    loss = cfg.TRAIN.DICE_WEIGHT * loss_dice + cfg.TRAIN.CE_WEIGHT * loss_ce
    return loss


def calc_coarse_visual_loss(visual_logits, masks, ce_loss, dice_loss, cfg, down_ratio=4):
    """
    Train visual anchor as a coarse targetness prior, not as a strong source-domain segmentor.
    """
    target_size = (cfg.DATASET.SIZE // down_ratio, cfg.DATASET.SIZE // down_ratio)

    if visual_logits.ndim == 3:
        visual_logits_4d = visual_logits.unsqueeze(1)
    else:
        visual_logits_4d = visual_logits

    if masks.ndim == 3:
        masks_4d = masks.unsqueeze(1).float()
    else:
        masks_4d = masks.float()

    visual_logits_low = F.interpolate(
        visual_logits_4d,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    masks_low = F.interpolate(
        masks_4d,
        size=target_size,
        mode="nearest",
    ).squeeze(1)

    return calc_loss(visual_logits_low, masks_low, ce_loss, dice_loss, cfg)


def soft_bce_logits(student_logits, teacher_logits):
    target = torch.sigmoid(teacher_logits.detach())
    return F.binary_cross_entropy_with_logits(student_logits, target)


def soft_dice_loss_from_logits(student_logits, teacher_logits, eps=1e-6):
    ps = torch.sigmoid(student_logits)
    pt = torch.sigmoid(teacher_logits.detach())
    dims = tuple(range(1, ps.ndim))
    inter = (ps * pt).sum(dim=dims)
    denom = ps.sum(dim=dims) + pt.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def distill_loss(student_logits, teacher_logits):
    return soft_bce_logits(student_logits, teacher_logits) + soft_dice_loss_from_logits(student_logits, teacher_logits)


def build_model_from_cfg(cfg):
    if cfg.MODEL.CLIP_MODEL == "unimedclip":
        return build_medclipseg_unimedclip(cfg)
    elif cfg.MODEL.CLIP_MODEL == "biomedclip":
        return build_medclipseg_biomedclip(cfg)
    elif cfg.MODEL.CLIP_MODEL == "clip":
        return build_medclipseg_clip(cfg)
    elif cfg.MODEL.CLIP_MODEL == "pubmedclip":
        return build_medclipseg_pubmedclip(cfg)
    else:
        raise ValueError(f"Unknown CLIP model: {cfg.MODEL.CLIP_MODEL}")


def canonical_source_name(name):
    n = str(name).lower()
    for suffix in ["_25", "_50", "_75", "_100"]:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
    if "busi" in n:
        return "BUSI"
    if "isic" in n:
        return "ISIC"
    if "kvasir" in n:
        return "Kvasir"
    if "btmri" in n or "brain" in n:
        return "BTMRI"
    raise ValueError(f"Cannot infer teacher source from DATASET.NAME={name}. Please set --teacher-source BTMRI/BUSI/ISIC/Kvasir.")


def resolve_teacher_ckpt(root, source_name):
    root = os.path.expanduser(root)
    aliases = {
        "BTMRI": ["BTMRI"],
        "BUSI": ["BUSI"],
        "ISIC": ["ISIC"],
        "Kvasir": ["Kvasir", "Kvasir-SEG", "KVASIR"],
    }
    names = aliases.get(source_name, [source_name])
    candidates = []
    for n in names:
        candidates.extend([
            os.path.join(root, n),
            os.path.join(root, f"{n}.pth"),
            os.path.join(root, f"{n}.pt"),
            os.path.join(root, f"{n}.ckpt"),
        ])
        d = os.path.join(root, n)
        if os.path.isdir(d):
            preferred_names = ["best_dice.pth", "best.pth", "model.pth", "checkpoint.pth", f"{n}.pth", f"{n}.pt"]
            for fn in preferred_names:
                candidates.append(os.path.join(d, fn))
            for fn in os.listdir(d):
                if fn.endswith((".pth", ".pt", ".ckpt")):
                    candidates.append(os.path.join(d, fn))
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Cannot find teacher checkpoint for source={source_name} under {root}.")


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def clean_state_dict_keys(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_weights(model, ckpt_path, strict=True, logger=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = clean_state_dict_keys(extract_state_dict(ckpt))
    if strict:
        model.load_state_dict(state_dict, strict=True)
        if logger is not None:
            logger.info(f"Strictly loaded checkpoint: {ckpt_path}")
    else:
        model_dict = model.state_dict()
        matched = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(k)
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=True)
        if logger is not None:
            logger.info(f"Partially loaded checkpoint: {ckpt_path}")
            logger.info(f"Matched params: {len(matched)}")
            logger.info(f"Skipped params: {len(skipped)}")
    return model


def evaluate_validation_loss(model, val_dataloader, device):
    """
    Use text-only validation to select checkpoints, so the main text branch is not harmed by visual fallback.
    """
    model.eval()
    val_losses = []
    dice_scores = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            images = batch["image"].to(device)
            masks = batch["ground_truth_mask"].to(device)
            text = batch["text_prompt"]

            seg_samples = model(
                images,
                text=text,
                num_samples=1,
                use_filter=True,
                return_visual=False,
                fuse_visual=False,
            )
            logits = seg_samples.mean(dim=0)

            loss = calc_loss(logits, masks, ce_loss, dice_loss, cfg)
            val_losses.append(loss.item())

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            if preds.ndim == 3:
                preds = preds.unsqueeze(1)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_scores.extend(dice.cpu().numpy())

    avg_loss = mean(val_losses)
    avg_dice = mean(dice_scores)
    model.train()
    return avg_loss, avg_dice


cfg = get_arguments()
source_dataset_name = cfg.teacher_source if cfg.teacher_source else cfg.DATASET.NAME
cfg.DATASET.NAME = cfg.DATASET.NAME + f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else cfg.DATASET.NAME

os.makedirs(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}"), exist_ok=True)
logger = logger_config(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", "log.txt"))
logger.info("************")
logger.info("** Config **")
logger.info("************")
logger.info(cfg)

if cfg.seed >= 0:
    logger.info("Setting fixed seed: {}".format(cfg.seed))
    set_random_seed(cfg.seed)


def worker_init_fn(worker_id):
    seed = cfg.seed + worker_id
    random.seed(seed)
    np.random.seed(seed)


ce_loss = BCEWithLogitsLoss()
dice_loss = monai.losses.DiceLoss(include_background=False, sigmoid=True, reduction="mean")

train_tf = transforms.Compose([RandomGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])])
val_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])

train_text_file = f"Train_text_{cfg.data_percentage}.xlsx" if cfg.data_percentage != 100 else "Train_text.xlsx"
val_text_file = f"Val_text_{cfg.data_percentage}.xlsx" if cfg.data_percentage != 100 else "Val_text.xlsx"
train_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + train_text_file)
val_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + val_text_file)

train_dataset = DatasetSegmentation(cfg.DATASET.TRAIN_PATH, cfg.DATASET.NAME, train_text, train_tf, image_size=cfg.DATASET.SIZE)
val_dataset = DatasetSegmentation(cfg.DATASET.VAL_PATH, cfg.DATASET.NAME, val_text, val_tf, image_size=cfg.DATASET.SIZE)

train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, worker_init_fn=worker_init_fn, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, worker_init_fn=worker_init_fn, num_workers=8, pin_memory=True)

if cfg.MODEL.CLIP_MODEL == "unimedclip":
    model = build_model_from_cfg(cfg)
    teacher_source = canonical_source_name(source_dataset_name)
    teacher_ckpt_path = resolve_teacher_ckpt(cfg.teacher_ckpt_root, teacher_source)
    logger.info(f"Teacher source dataset: {teacher_source}")
    logger.info(f"Teacher checkpoint: {teacher_ckpt_path}")

    if not cfg.no_init_student_from_teacher:
        model = load_weights(model, teacher_ckpt_path, strict=False, logger=logger)

    teacher = build_model_from_cfg(cfg)
    teacher = load_weights(teacher, teacher_ckpt_path, strict=False, logger=logger)
    teacher.to(cfg.MODEL.DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

elif cfg.MODEL.CLIP_MODEL == "biomedclip":
    model = build_medclipseg_biomedclip(cfg)
elif cfg.MODEL.CLIP_MODEL == "clip":
    model = build_medclipseg_clip(cfg)
elif cfg.MODEL.CLIP_MODEL == "pubmedclip":
    model = build_medclipseg_pubmedclip(cfg)
else:
    raise ValueError(f"Unknown CLIP model: {cfg.MODEL.CLIP_MODEL}")

enabled = set()
for name, param in model.named_parameters():
    if param.requires_grad:
        enabled.add(name)
logger.info(f"Parameters to be updated: {enabled}")
logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LEARNING_RATE)
num_epochs = cfg.TRAIN.NUM_EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
results_name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{backbone_name}"
resume_path = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", f"{results_name}_latest.pth")

start_epoch = 0
best_loss = float("inf")
best_dice = 0

if cfg.resume and os.path.exists(resume_path):
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint["model"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint.get("scheduler", {}))
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint.get("best_loss", float("inf"))
    best_dice = checkpoint.get("best_dice", 0)
    logger.info(f"Loaded checkpoint from epoch {start_epoch}, best loss: {best_loss:.4f}, best dice: {best_dice:.4f}")

model.train()
model.to(cfg.MODEL.DEVICE)

total_loss = []

for epoch in range(start_epoch, num_epochs):
    epoch_losses = []

    for i, batch in enumerate(tqdm(train_dataloader)):
        images = batch["image"].to(cfg.MODEL.DEVICE)
        masks = batch["ground_truth_mask"].to(cfg.MODEL.DEVICE)
        text = batch["text_prompt"]

        seg_logits, visual_logits, clip_loss = model(
            image=images,
            text=text,
            use_filter=True,
            return_visual=True,
        )

        with torch.no_grad():
            teacher_logits = teacher(
                images,
                text=text,
                num_samples=1,
                use_filter=False,
            )[0]

        loss_seg = calc_loss(seg_logits, masks, ce_loss, dice_loss, cfg)
        loss_visual = calc_coarse_visual_loss(
            visual_logits,
            masks,
            ce_loss,
            dice_loss,
            cfg,
            down_ratio=4,
        )
        loss_anchor = distill_loss(seg_logits, teacher_logits)

        # Only pull the visual prior toward the clean text prediction; do not let it corrupt the text branch.
        loss_cons = soft_dice_loss_from_logits(visual_logits, seg_logits.detach())

        lambda_anchor = 0.5
        lambda_visual = 0.25
        lambda_cons = 0.00

        loss = (
            loss_seg
            + cfg.TRAIN.CLIP_WEIGHT * clip_loss
            + lambda_anchor * loss_anchor
            + lambda_visual * loss_visual
            + lambda_cons * loss_cons
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    scheduler.step()
    mean_epoch_loss = mean(epoch_losses)
    mean_val_loss, mean_val_dice = evaluate_validation_loss(model, val_dataloader, cfg.MODEL.DEVICE)

    logger.info(f"EPOCH: {epoch + 1} | Training Loss: {mean_epoch_loss:.4f} | Validation Loss: {mean_val_loss:.4f}")

    if mean_val_dice > best_dice:
        logger.info(f"New best Dice: {best_dice:.4f} → {mean_val_dice:.4f}")
        best_dice = mean_val_dice
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice": best_dice,
            "best_loss": best_loss,
        }, os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", f"{results_name}_best_dice.pth"))
    else:
        logger.info(f"Dice: {mean_val_dice:.4f}")

    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_loss": best_loss,
        "best_dice": best_dice,
    }, os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", f"{results_name}_latest.pth"))
