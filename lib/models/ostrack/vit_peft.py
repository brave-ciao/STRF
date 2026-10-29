import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp, DropPath

from .utils import combine_tokens
from lib.models.layers.patch_embed import PatchEmbed
from .vit import VisionTransformer


class ThermalContrastEnhancer(nn.Module):

    def __init__(self, token_dim=768, contrast_history_length=5, gamma=0.7):
        super().__init__()
        self.contrast_history_length = contrast_history_length
        self.gamma = gamma

        self.contrast_net = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim // 2, token_dim),
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)

    def _temporal_weighted_avg(self, history_list):

        T = len(history_list)
        stack = torch.stack(history_list, dim=0)
        device = stack.device
        idx = torch.arange(T, device=device, dtype=torch.float32)

        weights = self.gamma ** (T - 1 - idx)
        weights = weights / (weights.sum() + 1e-6)
        w = weights.view(T, 1, 1, 1)
        return (stack * w).sum(dim=0)

    def forward(self, tokens, thermal_history=None):

        if thermal_history is not None and len(thermal_history) > 0:
            recent_history = thermal_history[-self.contrast_history_length:]
            historical_avg = self._temporal_weighted_avg(recent_history)

            contrast_weights = self.contrast_net(tokens)
        else:
            historical_avg = None
            contrast_weights = self.contrast_net(tokens)


        contrast_weights = contrast_weights + 1.0
        enhanced_tokens = tokens * contrast_weights

        return enhanced_tokens


class ThermalInterferenceSuppressor(nn.Module):

    def __init__(self, token_dim=768):
        super().__init__()

        self.spatial_suppressor = nn.Sequential(
            nn.Linear(token_dim * 2 + 1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, tokens, attn_weights):

        B, N, D = tokens.shape

        if attn_weights is None:

            return tokens


        attn = attn_weights.mean(dim=1)

        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)


        neighbor_feat = torch.matmul(attn, tokens)


        attn_importance = attn.mean(dim=-1, keepdim=True)

        attn_min = attn_importance.amin(dim=1, keepdim=True)
        attn_max = attn_importance.amax(dim=1, keepdim=True)
        attn_importance = (attn_importance - attn_min) / (attn_max - attn_min + 1e-6)

        context_input = torch.cat([tokens, neighbor_feat, attn_importance], dim=-1)
        base_suppress = self.spatial_suppressor(context_input)


        effective_suppress = base_suppress * (1.0 - attn_importance)
        effective_suppress = effective_suppress.clamp(0.0, 1.0)

        suppression_weights = 1.0 - effective_suppress
        return tokens * suppression_weights


class MultiTargetDiscriminator(nn.Module):

    def __init__(self, token_dim=768):
        super().__init__()
        self.verifier = nn.Sequential(
            nn.Linear(token_dim + 1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, current_tokens, target_template):

        B, N, D = current_tokens.shape
        device = current_tokens.device

        if target_template is None:
            return torch.ones(B, N, 1, device=device)


        target_mean = target_template.mean(dim=1, keepdim=True)
        target_mean = target_mean.expand(B, N, D)


        cur_norm = F.normalize(current_tokens, dim=-1)
        tgt_norm = F.normalize(target_mean, dim=-1)
        cos = F.cosine_similarity(cur_norm, tgt_norm, dim=-1, eps=1e-6)
        cos01 = (cos + 1.0) / 2.0
        cos01 = cos01.unsqueeze(-1)

        inp = torch.cat([current_tokens, cos01], dim=-1)
        scores = self.verifier(inp)
        return scores


class ThermalTrailHandler(nn.Module):

    def __init__(self, token_dim=768, trail_history_length=5, gamma=0.7):
        super().__init__()
        self.trail_history_length = trail_history_length
        self.gamma = gamma

        self.trail_detector = nn.Sequential(
            nn.Linear(token_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _temporal_weighted_avg(self, history_list):
        T = len(history_list)
        stack = torch.stack(history_list, dim=0)
        device = stack.device
        idx = torch.arange(T, device=device, dtype=torch.float32)
        weights = self.gamma ** (T - 1 - idx)
        weights = weights / (weights.sum() + 1e-6)
        w = weights.view(T, 1, 1, 1)
        return (stack * w).sum(dim=0)

    def forward(self, current_tokens, thermal_history):

        B, N, D = current_tokens.shape
        device = current_tokens.device

        if thermal_history is None or len(thermal_history) == 0:
            return torch.zeros(B, N, 1, device=device)

        recent_history = thermal_history[-self.trail_history_length:]
        historical_avg = self._temporal_weighted_avg(recent_history)

        trail_input = torch.cat([current_tokens, historical_avg], dim=-1)
        trail_scores = self.trail_detector(trail_input)
        return trail_scores


class InfraredAwareFusion(nn.Module):

    def __init__(self, token_dim=768, trail_history_length=5):
        super().__init__()
        self.contrast_enhancer = ThermalContrastEnhancer(token_dim)
        self.interference_suppressor = ThermalInterferenceSuppressor(token_dim)
        self.target_discriminator = MultiTargetDiscriminator(token_dim)
        self.trail_handler = ThermalTrailHandler(token_dim, trail_history_length)


        self.fusion_gate = nn.Sequential(
            nn.Linear(token_dim + 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, visual_tokens, attn_weights, thermal_history, target_template):

        B, N, D = visual_tokens.shape

        contrast_enhanced = self.contrast_enhancer(visual_tokens, thermal_history)
        interference_suppressed = self.interference_suppressor(visual_tokens, attn_weights)
        target_scores = self.target_discriminator(visual_tokens, target_template)
        trail_scores = self.trail_handler(visual_tokens, thermal_history)


        contrast_mean = contrast_enhanced.mean(dim=-1, keepdim=True)
        interference_mean = interference_suppressed.mean(dim=-1, keepdim=True)


        fusion_input = torch.cat([
            visual_tokens,
            contrast_mean,
            interference_mean,
            target_scores,
            trail_scores
        ], dim=-1)

        fusion_weights = self.fusion_gate(fusion_input)


        x_enh = (
            contrast_enhanced * 0.4 +
            interference_suppressed * 0.3 +
            visual_tokens* target_scores* 0.3
        )

        enhanced_tokens = (1.0 - fusion_weights) * visual_tokens + fusion_weights * x_enh

        return enhanced_tokens, fusion_weights


class FusionAuxLoss(nn.Module):

    def __init__(self, weight=1e-3):
        super().__init__()
        self.weight = weight

    def forward(self, fusion_weights):
        if self.weight <= 0:
            return fusion_weights.new_tensor(0.0)
        return self.weight * (fusion_weights ** 2).mean()


class UltraEnhancedInfraredAwareManager(nn.Module):

    def __init__(self, token_dim=768, fusion_layers=None,
                 history_length=8, trail_history_length=5,
                 aux_loss_weight=1e-3):
        super().__init__()
        self.fusion_layers = fusion_layers or [4, 5, 6, 7]
        self.fusion_modules = nn.ModuleDict()
        for layer_idx in self.fusion_layers:
            self.fusion_modules[f'layer_{layer_idx}'] = InfraredAwareFusion(
                token_dim=token_dim,
                trail_history_length=trail_history_length
            )

        self.thermal_histories = {layer: [] for layer in self.fusion_layers}
        self.target_template = None
        self.max_history_len = history_length

        self.fusion_aux_loss = FusionAuxLoss(weight=aux_loss_weight)

    def clear_history(self):
        for layer in self.thermal_histories:
            self.thermal_histories[layer] = []
        self.target_template = None

    def set_target_template(self, template_features):

        self.target_template = template_features

    def cache_current_features(self, layer_idx, features):

        if self.training or layer_idx not in self.thermal_histories:
            return
        hist = self.thermal_histories[layer_idx]
        hist.append(features.detach())
        if len(hist) > self.max_history_len:
            hist.pop(0)

    def forward(self, visual_tokens, layer_idx, attn_weights,
                packed_historical_features=None):

        if layer_idx not in self.fusion_layers:
            return visual_tokens, None, visual_tokens.new_tensor(0.0)


        if self.training:
            thermal_history = packed_historical_features or []
        else:
            thermal_history = self.thermal_histories[layer_idx]

        fusion_module = self.fusion_modules[f'layer_{layer_idx}']
        enhanced_tokens, fusion_weights = fusion_module(
            visual_tokens, attn_weights, thermal_history, self.target_template
        )

        aux_loss = self.fusion_aux_loss(fusion_weights)


        if not self.training:
            self.cache_current_features(layer_idx, enhanced_tokens)

        return enhanced_tokens, fusion_weights, aux_loss


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        if return_attention:
            return x_out, attn
        return x_out


class BlockWithAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., drop=0.,
                 drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, mlp_ratio=4.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        x_attn, attn_weights = self.attn(self.norm1(x), return_attention=True)
        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn_weights
        return x


class VisionTransformerP(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None,
                 new_patch_size=None, act_layer=None, weight_init='',
                 search_size=(256, 256), template_size=(64, 64),
                 fusion_layers=None, use_adaptive_infrared=True,
                 history_length=8, trail_history_length=5):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU


        if new_patch_size is None:
            new_patch_size = patch_size

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_drop = nn.Dropout(p=drop_rate)


        Hs, Ws = search_size
        new_P_Hs, new_P_Ws = Hs // new_patch_size, Ws // new_patch_size
        self.num_patches_search = new_P_Hs * new_P_Ws

        Ht, Wt = template_size
        new_P_Ht, new_P_Wt = Ht // new_patch_size, Wt // new_patch_size
        self.num_patches_template = new_P_Ht * new_P_Wt

        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            BlockWithAttention(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)


        self.use_adaptive_infrared = use_adaptive_infrared
        self.fusion_layers = fusion_layers or [4, 5, 6, 7]
        if self.use_adaptive_infrared:
            self.fusion_manager = UltraEnhancedInfraredAwareManager(
                token_dim=embed_dim,
                fusion_layers=self.fusion_layers,
                history_length=history_length,
                trail_history_length=trail_history_length
            )

        self.cat_mode = 'direct'
        self.init_weights(weight_init)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'fusion_manager'):
            self.fusion_manager.train(mode)
        return self

    def forward_features(self, z, x, search_bbox_xyxy, data, mask_z=None, mask_x=None, return_last_attn=False,
                         video_id=None, timestamp=1.0, is_new_sequence=False):
        B = x.shape[0]


        z_feat = self.patch_embed(z)
        x_feat = self.patch_embed(x)
        z_feat = z_feat + self.pos_embed_z
        x_feat = x_feat + self.pos_embed_x

        use_adaptive_infrared = self.use_adaptive_infrared and hasattr(self, 'fusion_manager')


        if use_adaptive_infrared and not self.training:
            if is_new_sequence:
                self.fusion_manager.clear_history()


        historical_features_per_layer = {layer: [] for layer in self.fusion_layers} if (use_adaptive_infrared and self.training and 'history_images' in data) else None

        if self.training and use_adaptive_infrared and 'history_images' in data:
            history_imgs_list = data['history_images']
            if history_imgs_list.numel() > 0:
                T_hist = history_imgs_list.shape[0]
                max_fusion_layer = max(self.fusion_layers) if self.fusion_layers else 0

                for i in range(T_hist):
                    hist_imgs = history_imgs_list[i]
                    hist_search_feat = self.patch_embed(hist_imgs) + self.pos_embed_x
                    hist_combined = combine_tokens(z_feat, hist_search_feat, mode=self.cat_mode)
                    hist_combined = self.pos_drop(hist_combined)

                    cur_hist = hist_combined
                    for layer_idx in range(max_fusion_layer + 1):
                        blk = self.blocks[layer_idx]
                        cur_hist, _ = blk(cur_hist, return_attention=True)

                        if layer_idx in self.fusion_layers:
                            hist_search_tokens = cur_hist[:, self.num_patches_template:]
                            historical_features_per_layer[layer_idx].append(hist_search_tokens.detach())


        x_combined = combine_tokens(z_feat, x_feat, mode=self.cat_mode)
        x_combined = self.pos_drop(x_combined)

        total_aux_loss = x_combined.new_tensor(0.0)

        for layer_idx, blk in enumerate(self.blocks):
            x_combined, attn_weights = blk(x_combined, return_attention=True)

            if use_adaptive_infrared:

                z_tokens = x_combined[:, :self.num_patches_template]
                x_tokens = x_combined[:, self.num_patches_template:]


                attn_search = attn_weights[:, :,
                                           self.num_patches_template:,
                                           self.num_patches_template:]


                self.fusion_manager.set_target_template(z_tokens)


                current_hist_feats = None
                if self.training and historical_features_per_layer is not None:
                    current_hist_feats = historical_features_per_layer.get(layer_idx, [])

                enhanced_x, _, aux_loss = self.fusion_manager(
                    x_tokens, layer_idx, attn_search,
                    packed_historical_features=current_hist_feats
                )
                total_aux_loss = total_aux_loss + aux_loss

                x_combined = torch.cat([z_tokens, enhanced_x], dim=1)


        x_out = self.norm(x_combined)

        aux_dict = {"attn": None}
        if use_adaptive_infrared:
            aux_dict["fusion_loss"] = total_aux_loss

        return x_out, aux_dict

    def forward(self, z, x, search_bbox_xyxy, data,
                ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None, return_last_attn=False,
                video_id=None, timestamp=1.0, is_new_sequence=False):
        x, aux_dict = self.forward_features(
            z, x, search_bbox_xyxy, data,
            return_last_attn=return_last_attn,
            video_id=video_id, timestamp=timestamp,
            is_new_sequence=is_new_sequence
        )
        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerP(**kwargs)
    if pretrained:
        if 'npz' in str(pretrained):
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + str(pretrained))
            if missing_keys:
                print('Missing keys:', missing_keys)
            if unexpected_keys:
                print('Unexpected keys:', unexpected_keys)
    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        **kwargs
    )
    return _create_vision_transformer(pretrained=pretrained, **model_kwargs)
