import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticFilter(nn.Module):
    def __init__(self, dim, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.proj = nn.Linear(dim, dim)
        # for p in self.proj.parameters():
        #     p.requires_grad_(False)

        self.temp = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.tau = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, vis, text, filter_percent=0.3, prev_score=None):
        vis_patch = vis[:, 1:, :]  # [B, 196, D]

        text_proj = self.proj(text)
        
        vis_patch = F.normalize(vis_patch, dim=-1)
        text_norm = F.normalize(text_proj, dim=-1)

        sim = torch.matmul(text_norm, vis_patch.transpose(1, 2))

        filter_k = max(1, int(vis_patch.shape[1] * filter_percent))
        score = sim.topk(k=filter_k, dim=-1).values.mean(dim=-1)

        if prev_score is not None:
            score = self.alpha * prev_score + (1 - self.alpha) * score


        score_mean = score.mean(dim=-1, keepdim=True)
        score_std = score.std(dim=-1, keepdim=True) + 1e-8
        score = (score - score_mean) / score_std

        gate = torch.sigmoid(self.temp * (score - self.tau)).unsqueeze(-1)  # [B,256,1]

      
        B, L, D = gate.shape
        cls_mask = torch.zeros(L, device=gate.device, dtype=torch.bool)
        cls_mask[0] = True
        sep_mask = torch.zeros(L, device=gate.device, dtype=torch.bool)
        sep_mask[-1] = True

        
        gate = gate.masked_fill(cls_mask.unsqueeze(0).unsqueeze(-1), 1.0)
        gate = gate.masked_fill(sep_mask.unsqueeze(0).unsqueeze(-1), 1.0)

       
        text_filtered = text * (1 - gate) + text_proj * gate

        # if self.training:
        #     print(f"Gate mean: {gate.mean().item():.3f}, std: {gate.std().item():.3f}, min: {gate.min().item():.3f}, max: {gate.max().item():.3f}")

        return text_filtered, score, gate