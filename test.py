import os
import cv2
import torch
import argparse
import logging
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.dataloader import DatasetSegmentation, ValGenerator
from trainers import *
from utils.main_utils import load_cfg_from_cfg_file, read_text, normalize
import matplotlib.pyplot as plt


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--prompt_design", type=str, default="original", help="Text prompt design.")
    parser.add_argument("--data_percentage", type=int, default=100, help="Percentage of data to use.")
    parser.add_argument("--source_dataset", type=str, help="source dataset name for loading trained model.")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="override",
        choices=["text", "visual", "override"],
        help="Evaluation mode: text, visual, or safe_fused.",
    )
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update({k: v for k, v in vars(args).items()})
    return cfg


def logger_config(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding="UTF-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def main():
    cfg = get_arguments()

    if cfg.seed >= 0:
        print(f"Setting fixed seed: {cfg.seed}")
        set_random_seed(cfg.seed)

    cfg.DATASET.NAME = cfg.DATASET.NAME + f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else cfg.DATASET.NAME
    results_root = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}")
    os.makedirs(results_root, exist_ok=True)
    logger = logger_config(os.path.join(results_root, f"log_{cfg.prompt_design}_{cfg.eval_mode}.txt"))

    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)

    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
    results_name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{backbone_name}"
    checkpoint_type = "latest" if cfg.TEST.USE_LATEST else "best_dice"
    checkpoint_path = os.path.join(
        cfg.output_dir,
        cfg.source_dataset if cfg.data_percentage == 100 else cfg.DATASET.NAME,
        "trained_models",
        f"seed{cfg.seed}",
        f"{results_name}_{checkpoint_type}.pth",
    )

    if cfg.MODEL.CLIP_MODEL == "unimedclip":
        model = build_medclipseg_unimedclip(cfg)
    elif cfg.MODEL.CLIP_MODEL == "biomedclip":
        model = build_medclipseg_biomedclip(cfg)
    elif cfg.MODEL.CLIP_MODEL == "clip":
        model = build_medclipseg_clip(cfg)
    elif cfg.MODEL.CLIP_MODEL == "pubmedclip":
        model = build_medclipseg_pubmedclip(cfg)
    else:
        raise ValueError(f"Unknown CLIP model: {cfg.MODEL.CLIP_MODEL}")

    checkpoint = torch.load(checkpoint_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval().to(cfg.MODEL.DEVICE)

    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    logger.info(f"Evaluation mode: {cfg.eval_mode}")

    test_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
    test_text_file = f"Test_text_{cfg.prompt_design}.xlsx"
    test_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + test_text_file)

    test_dataset = DatasetSegmentation(
        cfg.DATASET.TEST_PATH,
        cfg.DATASET.NAME,
        test_text,
        test_tf,
        image_size=cfg.DATASET.SIZE,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    save_dir = os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "seg_results",
        f"seed{cfg.seed}",
        results_name + f"_Prompt-{cfg.prompt_design}_{cfg.eval_mode}",
    )
    save_unc_dir = os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "unc_results",
        f"seed{cfg.seed}",
        results_name + f"_Prompt-{cfg.prompt_design}_{cfg.eval_mode}",
    )
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_unc_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            out = model(
                image=batch["image"].to(cfg.MODEL.DEVICE),
                text=batch["text_prompt"],
                num_samples=cfg.TEST.NUM_SAMPLES,
                use_filter=False,
                return_visual=True,
                fuse_visual=(cfg.eval_mode == "override"),
            )

            if cfg.eval_mode == "text":
                seg_samples = torch.sigmoid(out["seg_samples"])
                seg_logits = seg_samples.mean(dim=0)
            elif cfg.eval_mode == "visual":
                seg_logits = torch.sigmoid(out["visual_logits"])
            elif cfg.eval_mode == "override":
                seg_samples = torch.sigmoid(out["seg_samples"])
                seg_logits = seg_samples.mean(dim=0)
            else:
                raise ValueError(f"Unknown eval_mode: {cfg.eval_mode}")

            text_conf = out.get("text_conf", None)
            visual_conf = out.get("visual_conf", None)
            use_visual = out.get("use_visual", out.get("visual_weight", None))
            text_area = out.get("text_area", None)
            text_peak = out.get("text_peak", None)

            seg_unc = -(
                seg_logits * torch.log(seg_logits + 1e-8)
                + (1.0 - seg_logits) * torch.log(1.0 - seg_logits + 1e-8)
            )
            mask_preds = seg_logits > 0.5

            dataset_names = batch["dataset_name"]
            mask_names = batch["mask_name"]

            for i in range(len(dataset_names)):
                pred_mask = mask_preds[i].cpu().numpy().astype(np.uint8)
                dataset_name = dataset_names[i]
                mask_name = mask_names[i]

                binary_pred = np.uint8(pred_mask > 0)
                cv2.imwrite(os.path.join(save_dir, mask_name), binary_pred * 255)

                u_map = seg_unc[i].cpu().numpy()
                u_map = normalize(u_map)
                colormap = plt.get_cmap("nipy_spectral")
                u_map_color = (colormap(u_map)[:, :, :3] * 255).astype(np.uint8)
                u_map_colored = cv2.cvtColor(u_map_color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_unc_dir, mask_name), u_map_colored)

                if text_conf is not None:
                    logger.info(
                        f"{dataset_name} | {mask_name} | "
                        f"text_conf={text_conf[i].item():.4f} | "
                        f"visual_conf={visual_conf[i].item():.4f} | "
                        f"text_area={text_area[i].item():.6f} | "
                        f"text_peak={text_peak[i].item():.4f} | "
                        f"use_visual={use_visual[i].item():.4f}"
                    )

if __name__ == "__main__":
    main()
