import torch
import torch.nn as nn
from torch.nn import functional as F
from open_clip_lib import create_model_and_transforms, HFTokenizer, get_mean_std
from typing import Optional
from .layers import PVL_Adapter
from .scale_block import ScaleBlock
from .layers_filter import SemanticFilter
import os
from huggingface_hub import hf_hub_download

def download_checkpoint(filename: str):

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    local_path = os.path.join(ckpt_dir, filename)

    if os.path.isfile(local_path):
        print(f"Found checkpoint: {local_path}")
        return local_path

    print(f"Checkpoint not found. Downloading {filename} from Hugging Face...")

    hf_hub_download(
        repo_id="TahaKoleilat/MedCLIPSeg",
        repo_type="model",
        filename=f"checkpoints/{filename}",
        local_dir=".",
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded checkpoint to {local_path}")
    return local_path


def load_unimedclip_to_device(cfg):

    if cfg.MODEL.BACKBONE == "ViT-B/16":
        model_name = "ViT-B-16-quickgelu"
        pretrained_weights = download_checkpoint("unimed_clip_vit_b16.pt")

    elif cfg.MODEL.BACKBONE == "ViT-L/14":
        model_name = "ViT-L-14-336-quickgelu"
        pretrained_weights = download_checkpoint("unimed_clip_vit_l14_base_text_encoder.pt")

    else:
        raise NotImplementedError(f"Backbone {cfg.MODEL.BACKBONE} not implemented.")

    text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"

    mean, std = get_mean_std()
    device = cfg.MODEL.DEVICE

    model, _, _ = create_model_and_transforms(
        model_name,
        pretrained_weights,
        precision="amp",
        device=device,
        force_quick_gelu=True,
        mean=mean,
        std=std,
        inmem=True,
        text_encoder_name=text_encoder_name,
    )

    return model.to(device).eval()


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model, output_hidden_states=False):
        super(CustomCLIP, self).__init__()

        self.cfg = cfg
        self.vision_model = clip_model.visual
        self.text_model = clip_model.text_encoder
        self.logit_scale = clip_model.logit_scale
        self.temperature = cfg.MODEL.TEMPERATURE
        self.fusion_stages = cfg.MODEL.LAYERS

        if cfg.MODEL.BACKBONE == "ViT-B/16":
            self.embed_dim = 768
            self.patch_size = 16
            self.text_proj_dim = 512

        elif cfg.MODEL.BACKBONE == "ViT-L/14":
            self.embed_dim = 1024
            self.patch_size = 14
            self.text_proj_dim = 768
            raise NotImplementedError("ViT-L/14 not implemented yet.")

        self.output_hidden_states = output_hidden_states
        self.dtype = self.text_model.transformer.dtype
        self.im_size = cfg.DATASET.SIZE
        self.device = cfg.MODEL.DEVICE

        self.tokenizer = HFTokenizer(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            context_length=256,
            **{},
        )

        adapter_channels = cfg.MODEL.ADAPTER_DIM
        self.num_upscale = cfg.MODEL.NUM_UPSCALE
        self.beta = cfg.MODEL.BETA
        self.gate_init = cfg.MODEL.GATE_INIT

        self.mask_head = nn.Sequential(
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
        )

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.text_proj_dim) for _ in range(self.num_upscale)],
        )

        # ===== Visual Targetness Anchor branch =====
        # 这个分支不吃 text feature，用 image-only feature 学一个目标定位兜底 mask。
        self.visual_upscale = nn.Sequential(
            *[ScaleBlock(self.text_proj_dim) for _ in range(self.num_upscale)],
        )

        # self.visual_upscale = self.upscale
        hidden_dim = max(self.text_proj_dim // 4, 64)

        self.visual_anchor_head = nn.Sequential(
            nn.Conv2d(self.text_proj_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        self.pvl_adapters = nn.ModuleList([
            PVL_Adapter(
                in_channels_vis=self.embed_dim,
                in_channels_txt=self.embed_dim,
                adapter_channels=adapter_channels,
                beta=self.beta,
                gate_init=self.gate_init,
            )
            for _ in range(len(self.fusion_stages))
        ])

        self.semanticfilters = nn.ModuleList([
            SemanticFilter(dim=self.embed_dim, alpha=0.7)
            for _ in range(len(self.fusion_stages))
        ])

    def encode_image_only(self, image):
        x_img = self.vision_model.conv1(image)

        x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], -1)
        x_img = x_img.permute(0, 2, 1)

        cls_token = (
            self.vision_model.class_embedding.to(x_img.dtype)
            + torch.zeros(
                x_img.shape[0],
                1,
                x_img.shape[-1],
                dtype=x_img.dtype,
                device=x_img.device,
            )
        )

        x_img = torch.cat([cls_token, x_img], dim=1)
        x_img = x_img + self.vision_model.positional_embedding.to(x_img.dtype)
        x_img = self.vision_model.ln_pre(x_img)

        x_img = x_img.permute(1, 0, 2)

        for block in self.vision_model.transformer.resblocks:
            x_img = block(x_img)

        x_img = x_img.permute(1, 0, 2)
        x_img = self.vision_model.ln_post(x_img)

        if self.vision_model.proj is not None:
            x_img = x_img @ self.vision_model.proj

        return x_img

    def build_patch_map(self, image_features, B, H, W, upscale_module):
        seg_feats = image_features[:, 1:, :]
        seg_feats = seg_feats / (seg_feats.norm(dim=-1, keepdim=True) + 1e-6)

        h_patch = H // self.patch_size
        w_patch = W // self.patch_size

        seg_feats = seg_feats.reshape(B, h_patch, w_patch, -1)
        seg_feats = seg_feats.permute(0, 3, 1, 2)

        seg_feats = upscale_module(seg_feats)

        return seg_feats

    def encode_text_image(
        self,
        tokenized_prompts,
        text_prompts,
        image,
        attention_mask: Optional[torch.LongTensor] = None,
        use_filter: bool = False,
    ):

        if attention_mask is None:
            attention_mask = (
                tokenized_prompts != self.text_model.config.pad_token_id
            ).long()

        x_txt = self.text_model.transformer.embeddings(
            inputs_embeds=text_prompts
        )

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (
            1.0 - extended_attention_mask
        ) * torch.finfo(self.dtype).min

        x_img = self.vision_model.conv1(image)
        x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], -1)
        x_img = x_img.permute(0, 2, 1)

        x_img = torch.cat(
            [
                self.vision_model.class_embedding.to(x_img.dtype)
                + torch.zeros(
                    x_img.shape[0],
                    1,
                    x_img.shape[-1],
                    dtype=x_img.dtype,
                    device=x_img.device,
                ),
                x_img,
            ],
            dim=1,
        )

        x_img = x_img + self.vision_model.positional_embedding.to(x_img.dtype)
        x_img = self.vision_model.ln_pre(x_img)
        x_img = x_img.permute(1, 0, 2)

        hidden_states = []
        prev_score = None
        final_gate = None

        for i, (block, layer) in enumerate(
            zip(
                self.vision_model.transformer.resblocks,
                self.text_model.transformer.encoder.layer,
            )
        ):

            if i in self.fusion_stages:

                stage_idx = self.fusion_stages.index(i)

                # 只在增强分支开启 SemanticFilter
                # normal branch / teacher branch 默认 use_filter=False，不走这里
                if use_filter:
                    x_txt, score, gate = self.semanticfilters[self.fusion_stages.index(i)](
                        x_img.transpose(1,0), x_txt, filter_percent=0.3, prev_score=prev_score
                    )
                    prev_score = score
                    final_gate = gate

                vis_pvl, txt_pvl = self.pvl_adapters[stage_idx](
                    x_img.transpose(1, 0),
                    x_txt,
                )

                x_txt = x_txt + txt_pvl
                x_img = x_img + vis_pvl.transpose(1, 0)

            x_img = block(x_img)
            x_txt = layer(x_txt, attention_mask=extended_attention_mask)

            hidden_states.append(x_img)
            x_txt = x_txt[0]

        x_img = x_img.permute(1, 0, 2)

        x_img = self.vision_model.ln_post(x_img)

        if self.vision_model.proj is not None:
            x_img = x_img @ self.vision_model.proj

        pooled_out = x_txt[:, 0, :]
        projected = self.text_model.proj(pooled_out)
        x_txt = self.text_model.proj(x_txt)

        if self.output_hidden_states:
            return x_img, hidden_states, projected, x_txt, final_gate
        else:
            return x_img, projected, x_txt, final_gate


    def compute_seg_logits(self, image_features, text_features, B, H, W):
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

        cls_token = image_features[:, 0, :]
        cls_token = cls_token / (cls_token.norm(dim=-1, keepdim=True) + 1e-6)

        seg_feats = self.build_patch_map(
            image_features=image_features,
            B=B,
            H=H,
            W=W,
            upscale_module=self.upscale,
        )

        text_query = self.mask_head(text_features).unsqueeze(1)

        seg_logits = torch.einsum(
            "bqc,bchw->bqhw",
            text_query,
            seg_feats,
        )

        seg_logits = F.interpolate(
            seg_logits,
            self.im_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return seg_logits, cls_token

    def compute_visual_anchor_logits(self, image_features, B, H, W):
        """
        Text-agnostic visual targetness prediction.
        输出 [B, H, W]
        """
        visual_feats = self.build_patch_map(
            image_features=image_features,
            B=B,
            H=H,
            W=W,
            upscale_module=self.visual_upscale,
        )

        visual_logits = self.visual_anchor_head(visual_feats)

        visual_logits = F.interpolate(
            visual_logits,
            self.im_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return visual_logits

    def soft_dice_score(self, p1, p2, eps=1e-6):
        dims = tuple(range(1, p1.ndim))
        inter = (p1 * p2).sum(dim=dims)
        denom = p1.sum(dim=dims) + p2.sum(dim=dims)
        dice = (2.0 * inter + eps) / (denom + eps)
        return dice

    def compute_text_image_alignment(self, text_features, image_features):
        """
        image_features should come from encode_image_only(image), so this alignment
        is not affected by the noisy prompt through PVL adapters.
        """
        text_features = text_features / (
            text_features.norm(dim=-1, keepdim=True) + 1e-6
        )
        patch_feats = image_features[:, 1:, :]
        patch_feats = patch_feats / (
            patch_feats.norm(dim=-1, keepdim=True) + 1e-6
        )
        sim = torch.einsum("bd,bnd->bn", text_features, patch_feats)
        k = max(1, int(sim.shape[1] * 0.10))
        align = sim.topk(k=k, dim=1).values.mean(dim=1)
        return align

    def compute_text_confidence(
        self,
        text_logits,
        visual_logits,
        text_features,
        image_features_visual,
    ):
        """
        c_t in [0, 1]. Higher means the text-guided branch is more reliable.
        """
        p_text = torch.sigmoid(text_logits)
        p_visual = torch.sigmoid(visual_logits)

        # 1) Is there a high-confidence text response?
        flat = p_text.flatten(1)
        k = max(1, int(flat.shape[1] * 0.01))
        text_peak = flat.topk(k=k, dim=1).values.mean(dim=1)
        peak_conf = torch.sigmoid((text_peak - 0.35) * 12.0)

        # 2) Is the text mask almost empty?
        text_area = (p_text > 0.5).float().mean(dim=(1, 2))
        area_conf = torch.sigmoid((text_area - 0.002) * 80.0)

        # 3) Does the text embedding align with image-only patch features?
        align = self.compute_text_image_alignment(
            text_features=text_features,
            image_features=image_features_visual,
        )
        align_conf = torch.sigmoid((align - 0.15) * 12.0)

        # 4) Does the text mask roughly agree with the visual targetness map?
        agree = self.soft_dice_score(p_text, p_visual)
        agree_conf = torch.sigmoid((agree - 0.15) * 10.0)

        # Keep agreement weight small because visual targetness may be imperfect on target domains.
        text_conf = (
            0.45 * peak_conf
            + 0.25 * area_conf
            + 0.20 * align_conf
            + 0.10 * agree_conf
        )
        return text_conf.clamp(0.0, 1.0)

    def compute_visual_confidence(self, visual_logits):
        """
        c_v in [0, 1]. Higher means the visual anchor itself looks reliable.
        This prevents a source-trained visual branch from dominating on target domains.
        """
        p_visual = torch.sigmoid(visual_logits)

        # 1) Does the visual anchor have a clear response?
        flat = p_visual.flatten(1)
        k = max(1, int(flat.shape[1] * 0.01))
        visual_peak = flat.topk(k=k, dim=1).values.mean(dim=1)
        peak_conf = torch.sigmoid((visual_peak - 0.35) * 12.0)

        # 2) Is the visual area plausible?
        area = (p_visual > 0.5).float().mean(dim=(1, 2))
        area_low = 0.002
        area_high = 0.50
        low_conf = torch.sigmoid((area - area_low) * 80.0)
        high_conf = torch.sigmoid((area_high - area) * 20.0)
        area_conf = low_conf * high_conf

        # 3) Is the visual map not globally uncertain around 0.5?
        entropy = -(
            p_visual * torch.log(p_visual + 1e-8)
            + (1.0 - p_visual) * torch.log(1.0 - p_visual + 1e-8)
        )
        entropy = entropy.mean(dim=(1, 2)) / 0.69314718
        entropy_conf = torch.sigmoid((0.65 - entropy) * 8.0)

        visual_conf = (
            0.45 * peak_conf
            + 0.35 * area_conf
            + 0.20 * entropy_conf
        )
        return visual_conf.clamp(0.0, 1.0)

    def safe_fuse_with_visual_anchor(
        self,
        text_logits,
        visual_logits,
        text_features,
        image_features_visual,
    ):
        """
        Emergency Visual Override.

        Use text prediction by default. Only when the text branch is highly unreliable
        and the visual anchor is confident, fully switch to the visual prediction:

            if text_conf < tau_text_fail and visual_conf > tau_visual_reliable:
                P_final = P_visual
            else:
                P_final = P_text
        """
        p_text = torch.sigmoid(text_logits)
        p_visual = torch.sigmoid(visual_logits)
        B = p_text.shape[0]

        text_conf = self.compute_text_confidence(
            text_logits=text_logits,
            visual_logits=visual_logits,
            text_features=text_features,
            image_features_visual=image_features_visual,
        )
        visual_conf = self.compute_visual_confidence(visual_logits)

        # ===== Emergency Visual Override: only when text predicts almost nothing =====
        # text_area: proportion of pixels predicted as foreground by text branch
        text_area = (p_text > 0.5).float().mean(dim=(1, 2))

        # text_peak: mean probability of top 1% most confident pixels
        flat_text = p_text.flatten(1)
        k = max(1, int(flat_text.shape[1] * 0.01))
        text_peak = flat_text.topk(k=k, dim=1).values.mean(dim=1)

        # Conservative thresholds
        # For 224x224, text_area < 0.001 means fewer than about 50 foreground pixels.
        text_almost_empty = text_area < 0.0005
        text_very_weak = text_peak < 0.25
        visual_reliable = visual_conf > 0.80

        # Only switch to visual when text almost detects nothing.
        use_visual = text_almost_empty & text_very_weak & visual_reliable

        use_visual_map = use_visual.view(B, 1, 1)

        p_final = torch.where(
            use_visual_map,
            p_visual,
            p_text,
        )

        fused_logits = torch.logit(p_final.clamp(1e-5, 1.0 - 1e-5))

        aux = {
            "text_conf": text_conf,
            "visual_conf": visual_conf,
            "visual_weight": use_visual.float(),
            "use_visual": use_visual.float(),
            "text_area": text_area,
            "text_peak": text_peak,
        }

        return fused_logits, aux

    def soft_cross_entropy(self, pred_logits, soft_targets):
        log_probs = F.log_softmax(pred_logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss

    def forward(
        self,
        image,
        text,
        num_samples=30,
        use_filter=False,
        return_visual=False,
        fuse_visual=False,
    ):
        B, C, H, W = image.shape
        tokenized_prompts = self.tokenizer(text).to(self.device)

        with torch.no_grad():
            prompts = self.text_model.transformer.embeddings.word_embeddings(
                tokenized_prompts
            ).type(self.dtype)

        if self.training:
            image_features_text, text_features, dense_text_feat, gate_values = (
                self.encode_text_image(
                    tokenized_prompts,
                    prompts,
                    image,
                    use_filter=use_filter,
                )
            )

            text_logits, cls_token = self.compute_seg_logits(
                image_features_text,
                text_features,
                B,
                H,
                W,
            )

            visual_logits = None
            if return_visual:
                image_features_visual = self.encode_image_only(image)
                visual_logits = self.compute_visual_anchor_logits(
                    image_features_visual,
                    B,
                    H,
                    W,
                )

            patch_logits = image_features_text[:, 1:, :]
            patch_logits = patch_logits / (
                patch_logits.norm(dim=-1, keepdim=True) + 1e-6
            )
            patch_mean = patch_logits.mean(dim=1)

            logits_per_image = (patch_mean @ text_features.T) / self.temperature
            logits_per_text = (text_features @ patch_mean.T) / self.temperature

            with torch.no_grad():
                text_sim = (text_features @ text_features.T) / self.temperature
                text_sim = text_sim / (
                    text_sim.norm(dim=-1, keepdim=True) + 1e-6
                )
                soft_targets = F.softmax(text_sim, dim=-1)

            loss_i2t = self.soft_cross_entropy(logits_per_image, soft_targets)
            loss_t2i = self.soft_cross_entropy(logits_per_text, soft_targets.T)
            clip_loss = (loss_i2t + loss_t2i) / 2

            if return_visual:
                return text_logits, visual_logits, clip_loss
            return text_logits, clip_loss

        seg_samples = []
        aux_list = []

        image_features_visual = None
        visual_logits = None
        if return_visual or fuse_visual:
            image_features_visual = self.encode_image_only(image)
            visual_logits = self.compute_visual_anchor_logits(
                image_features_visual,
                B,
                H,
                W,
            )

        for _ in range(num_samples):
            image_features_text, text_features, _, _ = self.encode_text_image(
                tokenized_prompts,
                prompts,
                image,
                use_filter=use_filter,
            )

            text_logits, _ = self.compute_seg_logits(
                image_features_text,
                text_features,
                B,
                H,
                W,
            )

            if fuse_visual:
                fused_logits, aux = self.safe_fuse_with_visual_anchor(
                    text_logits=text_logits,
                    visual_logits=visual_logits,
                    text_features=text_features,
                    image_features_visual=image_features_visual,
                )
                seg_samples.append(fused_logits)
                aux_list.append(aux)
            else:
                seg_samples.append(text_logits)

        seg_samples = torch.stack(seg_samples, dim=0)

        if return_visual:
            out = {
                "seg_samples": seg_samples,
                "visual_logits": visual_logits,
            }
            if len(aux_list) > 0:
                out["text_conf"] = torch.stack(
                    [x["text_conf"] for x in aux_list], dim=0
                ).mean(dim=0)
                out["visual_conf"] = torch.stack(
                    [x["visual_conf"] for x in aux_list], dim=0
                ).mean(dim=0)
                out["visual_weight"] = torch.stack(
                    [x["visual_weight"] for x in aux_list], dim=0
                ).mean(dim=0)
                out["use_visual"] = torch.stack(
                    [x["use_visual"] for x in aux_list], dim=0
                ).mean(dim=0)

                out["text_area"] = torch.stack(
                    [x["text_area"] for x in aux_list], dim=0
                ).mean(dim=0)

                out["text_peak"] = torch.stack(
                    [x["text_peak"] for x in aux_list], dim=0
                ).mean(dim=0)
            return out

        return seg_samples


def build_medclipseg_unimedclip(cfg):
    print(f"Loading UniMedCLIP (backbone: {cfg.MODEL.BACKBONE})")
    clip_model = load_unimedclip_to_device(cfg)
    clip_model.float()

    print("Building custom UniMedCLIP")
    model = CustomCLIP(cfg, clip_model)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "pvl_adapters" in name:
            param.requires_grad_(True)
        elif "mask_head" in name:
            param.requires_grad_(True)
        elif "upscale" in name:
            param.requires_grad_(True)
        elif "visual_upscale" in name:
            param.requires_grad_(True)
        elif "visual_anchor_head" in name:
            param.requires_grad_(True)
        # elif "semanticfilters" in name:
        #     param.requires_grad_(
        #         name.endswith(".temp") or name.endswith(".tau")
        #     )
        else:
            param.requires_grad_(False)

    return model
