import wandb
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmpose.models.builder import build_loss
from timm.models.layers import trunc_normal_

wandb.init(
    project="pose_estimation",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "Human_Pose_Estimation",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

def tokenizer(drop_rate=0.1):
    def __init__(self):
        super().__init__()
        self.start_embed = nn.Linear(2, int(self.enc_hidden_dim*(1-self.guide_ratio)))
        self.encoder = nn.ModuleList(
            [MixerLayer(self.enc_hidden_dim, self.enc_hidden_inter_dim, 
                self.num_joints, self.enc_token_inter_dim,
                self.enc_dropout) for _ in range(self.enc_num_blocks)])
        self.token_mlp = nn.Linear(self.num_joints, self.token_num)
        self.feature_embed = nn.Linear(self.enc_hidden_dim, self.token_dim)
        self.encoder_layer_norm = nn.LayerNorm(self.enc_hidden_dim)

        self.register_buffer('codebook', 
            torch.empty(self.token_class_num, self.token_dim))
        self.codebook.data.normal_()
        self.register_buffer('ema_cluster_size', 
            torch.zeros(self.token_class_num))
        self.register_buffer('ema_w', 
            torch.empty(self.token_class_num, self.token_dim))
        self.ema_w.data.normal_()

        self.decoder_token_mlp = nn.Linear(
        self.token_num, self.num_joints)
        self.decoder_start = nn.Linear(
            self.token_dim, self.dec_hidden_dim)
        self.decoder = nn.ModuleList(
            [MixerLayer(self.dec_hidden_dim, self.dec_hidden_inter_dim,
                self.num_joints, self.dec_token_inter_dim, 
                self.dec_dropout) for _ in range(self.dec_num_blocks)])
        self.decoder_layer_norm = nn.LayerNorm(self.dec_hidden_dim)

        self.recover_embed = nn.Linear(self.dec_hidden_dim, 2)
        self.loss = build_loss(tokenizer['loss_keypoint'])

    def forward(self, joints):
        joints_coord, joints_visible, bs = joints[:,:,:-1], joints[:,:,-1].bool(), joints.shape[0]
        encode_feat = self.start_embed(joints_coord)
        rand_mask_ind = torch.rand(joints_visible.shape) > self.drop_rate
        joints_visible = torch.logical_and(rand_mask_ind, joints_visible) 
        mask_tokens = self.invisible_token.expand(bs, joints.shape[1], -1)
        w = joints_visible.unsqueeze(-1).type_as(mask_tokens)
        encode_feat = encode_feat * w + mask_tokens * (1 - w)
        for num_layer in self.encoder:
            encode_feat = num_layer(encode_feat)
        encode_feat = self.encoder_layer_norm(encode_feat)
        encode_feat = encode_feat.transpose(2, 1)
        encode_feat = self.token_mlp(encode_feat).transpose(2, 1)
        encode_feat = self.feature_embed(encode_feat).flatten(0,1)
        distances = torch.sum(encode_feat**2, dim=1, keepdim=True) + torch.sum(self.codebook**2, dim=1) - 2 * torch.matmul(encode_feat, self.codebook.t())
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.shape[0], self.token_class_num, device=joints.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        dw = torch.matmul(encodings.t(), encode_feat.detach())
        combined = torch.cat((encodings.flatten(), dw.flatten()))
        sync_encodings, sync_dw = torch.split(combined, [n_encodings, n_dw])
        self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(sync_encodings, 0)
        n = torch.sum(self.ema_cluster_size.data)
        self.ema_cluster_size = ((self.ema_cluster_size + 1e-5) / (n + self.token_class_num * 1e-5) * n)
        self.ema_w = self.ema_w * self.decay + (1 - self.decay) * sync_dw
        self.codebook = self.ema_w / self.ema_cluster_size.unsqueeze(1)     
        e_latent_loss = F.mse_loss(part_token_feat.detach(), encode_feat)
        part_token_feat = encode_feat + (part_token_feat - encode_feat).detach()

        part_token_feat = part_token_feat.view(bs, -1, self.token_dim)
        part_token_feat = part_token_feat.transpose(2,1)
        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2,1)
        decode_feat = self.decoder_start(part_token_feat)
        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)
        recoverd_joints = self.recover_embed(decode_feat)

        part_token_feat = part_token_feat.view(bs, -1, self.token_dim)
        part_token_feat = part_token_feat.transpose(2,1)
        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2,1)
        decode_feat = self.decoder_start(part_token_feat)
        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)
        recoverd_joints = self.recover_embed(decode_feat)

class MixerLayer(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 hidden_inter_dim, 
                 token_dim, 
                 token_inter_dim, 
                 dropout_ratio):
        super().__init__()
        
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.MLP_token = MLPBlock(token_dim, token_inter_dim, dropout_ratio)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.MLP_channel = MLPBlock(hidden_dim, hidden_inter_dim, dropout_ratio)

    def forward(self, x):
        y = self.layernorm1(x)
        y = y.transpose(2, 1)
        y = self.MLP_token(y)
        y = y.transpose(2, 1)
        z = self.layernorm2(x + y)
        z = self.MLP_channel(z)
        out = x + y + z
        return out

class MLPBlock(nn.Module):
    def __init__(self, dim, inter_dim, dropout_ratio):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, inter_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(inter_dim, dim),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x):
        return self.ff(x)
    
wandb.log({"acc": acc, "loss": loss})
wandb.finish()