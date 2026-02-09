# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import numpy as np
import dgl
from dgl.nn import GATv2Conv


class StaticAdjacencyGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, d_state=8, d_conv=2, expand=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim

    def forward(self, historical_flow, adj_mx_tensor=None, batch_idx=None, total_batches=None, epoch=None):
        if adj_mx_tensor is not None:
            batch_size = historical_flow.shape[0]
            static_adj = adj_mx_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            return static_adj
        else:
            batch_size = historical_flow.shape[0]
            identity_adj = torch.eye(self.num_nodes, device=historical_flow.device)
            return identity_adj.unsqueeze(0).expand(batch_size, -1, -1)

    def get_l1_regularization(self):
        return torch.tensor(0.0, device=next(iter(self.parameters())).device 
                           if len(list(self.parameters())) > 0 else 'cpu')


class AdaptiveMaskedSpatialTransformer(nn.Module):
    def __init__(self, d_model, num_nodes, dropout=0.1, k_hop=2,
                 mask_threshold=0.5, feature_weight=0.5, enhanced_pe=True):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.k_hop = k_hop
        self.mask_threshold = mask_threshold
        self.feature_weight = feature_weight
        self.enhanced_pe = enhanced_pe

        self.pos_embed = nn.Parameter(torch.randn(num_nodes, d_model) * 0.02)

        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.attn_out_proj = nn.Linear(d_model, d_model)

        self.mask_generator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.scale = d_model ** -0.5

    def _compute_k_hop_adjacency(self, adj_matrix, k):
        device = adj_matrix.device
        N = adj_matrix.shape[0]

        k_hop_adj = torch.eye(N, device=device)
        current_power = adj_matrix

        for i in range(1, k + 1):
            k_hop_adj = torch.maximum(k_hop_adj, (current_power > 0).float())
            if i < k:
                current_power = torch.matmul(current_power, adj_matrix)

        return k_hop_adj

    def _create_simple_positional_encoding(self, adj_matrix, x, device):
        num_nodes = adj_matrix.shape[0]
        pe = torch.zeros(num_nodes, self.d_model, device=device)
        position = torch.arange(0, num_nodes, dtype=torch.float, device=device).unsqueeze(1)

        half_dim = self.d_model // 2
        div_term = torch.exp(torch.arange(0, half_dim, dtype=torch.float, device=device) *
                             (-math.log(10000.0) / half_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if self.d_model % 2 == 1:
            pe[:, -1] = torch.sin(position[:, 0] * div_term[-1])

        if self.enhanced_pe:
            degree = adj_matrix.sum(dim=1, keepdim=True)
            degree_norm = degree / (degree.max() + 1e-8)
            degree_embed = degree_norm.repeat(1, self.d_model) * 0.05
            final_pe = pe + degree_embed
        else:
            final_pe = pe

        return final_pe

    def _create_adaptive_mask(self, x, adj_matrix, batch_size):
        k_hop_mask = self._compute_k_hop_adjacency(adj_matrix, self.k_hop)

        x_norm = F.normalize(x, dim=-1)
        similarity = torch.matmul(x_norm, x_norm.transpose(-2, -1))

        node_importance = self.mask_generator(x)
        adaptive_threshold = torch.quantile(node_importance.squeeze(-1), 0.3, dim=-1, keepdim=True).unsqueeze(-1)

        feature_mask = (similarity > adaptive_threshold).float()

        k_hop_mask_batch = k_hop_mask.unsqueeze(0).expand(batch_size, -1, -1)
        final_mask = k_hop_mask_batch + self.feature_weight * feature_mask
        final_mask = (final_mask > self.mask_threshold).float()

        return final_mask

    def forward(self, x, adj_matrix=None):
        batch_size, num_nodes, d_model = x.shape

        if adj_matrix is not None:
            pos_enc = self._create_simple_positional_encoding(adj_matrix, x, x.device)
        else:
            pos_enc = self.pos_embed

        pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)
        x_pos = x + pos_enc

        qkv = self.qkv_proj(x_pos)
        q, k, v = qkv.chunk(3, dim=-1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if adj_matrix is not None:
            adaptive_mask = self._create_adaptive_mask(x_pos, adj_matrix, batch_size)
            attn_scores = attn_scores.masked_fill(adaptive_mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = self.attn_out_proj(attn_output)

        output = self.norm(attn_output + x)
        return output, attn_weights


class AmTGBiMGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.1, K=3, num_nodes=207,
                 fusion_gate_dropout=0.05):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.K = K
        self.fusion_gate_dropout = fusion_gate_dropout

        assert out_channels % num_heads == 0
        self.per_head_dim = out_channels // num_heads

        self.gat = GATv2Conv(
            in_feats=in_channels,
            out_feats=self.per_head_dim,
            num_heads=num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            residual=True,
            allow_zero_in_degree=True
        )

        self.spatial_transformer = AdaptiveMaskedSpatialTransformer(
            d_model=out_channels,
            num_nodes=num_nodes,
            dropout=dropout,
            k_hop=2,
            mask_threshold=0.5,
            feature_weight=0.5
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Dropout(fusion_gate_dropout),
            nn.Sigmoid()
        )

        self.input_projection = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.output_projection = nn.Linear(out_channels, in_channels)

        self.gat_output_adjustment = nn.Linear(out_channels, out_channels)

        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)

    def create_efficient_batch_graph(self, dynamic_adj, device):
        batch_size, num_nodes, _ = dynamic_adj.shape
        all_edges_src = []
        all_edges_dst = []
        all_edge_weights = []
        node_offset = 0

        for i in range(batch_size):
            edge_indices = dynamic_adj[i].nonzero(as_tuple=False)
            if edge_indices.size(0) > 0:
                src = edge_indices[:, 0] + node_offset
                dst = edge_indices[:, 1] + node_offset
                weights = dynamic_adj[i][edge_indices[:, 0], edge_indices[:, 1]]
                valid_mask = weights > 1e-6
                if valid_mask.sum() > 0:
                    all_edges_src.append(src[valid_mask])
                    all_edges_dst.append(dst[valid_mask])
                    all_edge_weights.append(weights[valid_mask])
            node_offset += num_nodes

        if all_edges_src:
            all_src = torch.cat(all_edges_src, dim=0)
            all_dst = torch.cat(all_edges_dst, dim=0)
            all_weights = torch.cat(all_edge_weights, dim=0)
        else:
            all_src = torch.arange(batch_size * num_nodes, device=device)
            all_dst = torch.arange(batch_size * num_nodes, device=device)
            all_weights = torch.ones(batch_size * num_nodes, device=device) * 1e-3

        batched_g = dgl.graph((all_src, all_dst), num_nodes=batch_size * num_nodes).to(device)
        batched_g.edata['weight'] = all_weights
        return batched_g

    def forward(self, g, x, dynamic_adj=None):
        batch_size, num_nodes, num_timesteps = x.shape

        static_adj = None
        if dynamic_adj is not None:
            static_adj = dynamic_adj[0].detach()

        if dynamic_adj is not None:
            batched_g = self.create_efficient_batch_graph(dynamic_adj, x.device)
        else:
            batched_g = g

        x_flat = x.view(-1, num_timesteps)
        x_reshaped = x

        try:
            gat_raw_output = self.gat(batched_g, x_flat)
            if isinstance(gat_raw_output, tuple):
                gat_out = gat_raw_output[0]
            else:
                gat_out = gat_raw_output

            if gat_out.dim() == 3:
                gat_out = gat_out.view(gat_out.shape[0], -1)

            if gat_out.shape[1] != self.out_channels:
                gat_out = self.gat_output_adjustment(gat_out)

            gat_out = F.relu(gat_out)
            gat_out = gat_out.view(batch_size, num_nodes, self.out_channels)

            x_for_transformer = self.input_projection(x_reshaped)
            if x_for_transformer.shape[-1] != self.out_channels:
                temp_proj = nn.Linear(x_for_transformer.shape[-1], self.out_channels).to(x_for_transformer.device)
                x_for_transformer = temp_proj(x_for_transformer)

            spatial_out, attn_weights = self.spatial_transformer(x_for_transformer, static_adj)

            if gat_out.shape != spatial_out.shape:
                min_batch = min(gat_out.shape[0], spatial_out.shape[0])
                min_nodes = min(gat_out.shape[1], spatial_out.shape[1])
                min_features = min(gat_out.shape[2], spatial_out.shape[2])
                gat_out = gat_out[:min_batch, :min_nodes, :min_features]
                spatial_out = spatial_out[:min_batch, :min_nodes, :min_features]

            concat_features = torch.cat([gat_out, spatial_out], dim=-1)
            gate_weights = self.fusion_gate(concat_features)
            fused_out = gate_weights * spatial_out + (1 - gate_weights) * gat_out

            final_out = self.output_projection(fused_out)
            final_out = self.dropout(final_out)

        except Exception:
            x_proj = self.input_projection(x_reshaped)
            x_proj = F.relu(x_proj)
            final_out = self.output_projection(x_proj)
            final_out = self.dropout(final_out)

        if final_out.shape != x.shape:
            if final_out.numel() == x.numel():
                final_out = final_out.view(x.shape)
            else:
                final_out = x

        try:
            output = self.norm(final_out + x)
        except Exception:
            output = self.norm(final_out)

        return output


class AmTGBiMBlock(nn.Module):
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, len_input, adj_mx,
                 attention_heads=4, mamba_d_state='auto', mamba_d_conv=2, mamba_expand=1,
                 k_hop=2, mask_threshold=0.5, feature_weight=0.5, enhanced_pe=True,
                 fusion_gate_dropout=0.05):
        super().__init__()

        num_nodes = adj_mx.shape[0] if isinstance(adj_mx, np.ndarray) else adj_mx.shape[0]
        if mamba_d_state == 'auto':
            if num_nodes > 500:
                d_state = 16
            elif num_nodes > 100:
                d_state = 16
            else:
                d_state = 4
        else:
            d_state = int(mamba_d_state)

        self.enc_embedding = DataEmbedding_inverted(len_input, 256, 0.1)

        self.graph_conv = AmTGBiMGATLayer(
            in_channels=len_input,
            out_channels=nb_chev_filter,
            num_heads=attention_heads,
            dropout=0.1,
            K=K,
            num_nodes=num_nodes,
            fusion_gate_dropout=fusion_gate_dropout
        )

        self.graph_conv.spatial_transformer = AdaptiveMaskedSpatialTransformer(
            d_model=nb_chev_filter,
            num_nodes=num_nodes,
            dropout=0.1,
            k_hop=k_hop,
            mask_threshold=mask_threshold,
            feature_weight=feature_weight,
            enhanced_pe=enhanced_pe
        )

        self.adj_mx = adj_mx

        self.mbda = StaticAdjacencyGenerator(
            input_dim=in_channels,
            hidden_dim=64,
            num_nodes=num_nodes
        )

        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

        encoder_layers = []
        for _ in range(2):
            encoder_layers.append(
                EncoderLayer(
                    Mamba(d_model=256, d_state=d_state, d_conv=mamba_d_conv, expand=mamba_expand),
                    Mamba(d_model=256, d_state=d_state, d_conv=mamba_d_conv, expand=mamba_expand),
                    256,
                    1024,
                    dropout=0.1,
                    activation='gelu'
                )
            )

        self.encoder = Encoder(
            encoder_layers,
            norm_layer=torch.nn.LayerNorm(256)
        )

        self.projector = nn.Linear(256, len_input, bias=True)
        self.projector1 = nn.Linear(nb_time_filter, in_channels, bias=True)
        self.projector2 = nn.Linear(in_channels, nb_time_filter, bias=True)
        self.device = DEVICE

    def forward(self, x, batch_idx=None, total_batches=None, epoch=None):
        x0 = x
        if x0.dim() < 4:
            x0 = x0.permute(0, 2, 1).contiguous()
            x0 = torch.unsqueeze(x0, dim=2)

        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x0.shape
        x1 = torch.squeeze(x0, dim=2).contiguous().to(self.device)

        adj_mx_tensor = torch.from_numpy(self.adj_mx).to(self.device) if isinstance(self.adj_mx, np.ndarray) else self.adj_mx

        static_adj = self.mbda(x0, adj_mx_tensor, batch_idx, total_batches, epoch).to(self.device)

        edge_index = adj_mx_tensor.nonzero(as_tuple=False).T
        g = dgl.graph((edge_index[0], edge_index[1])).to(self.device)

        output_gcn = self.graph_conv(g, x1, static_adj)

        enc_out = self.enc_embedding(output_gcn)
        mamba_output, ATT = self.encoder(enc_out)
        mamba_output = self.projector(mamba_output).permute(0, 2, 1).contiguous()[:, :, :num_of_vertices]
        mamba_output = mamba_output.permute(0, 2, 1).contiguous()
        mamba_output = torch.unsqueeze(mamba_output, dim=3)
        mamba_output = self.projector2(mamba_output)

        x_residual = self.residual_conv(x0.permute(0, 2, 1, 3).contiguous())
        x_residual = x_residual.permute(0, 2, 3, 1).contiguous()
        x_residual1 = self.ln(F.relu(x_residual + mamba_output))
        x_residual2 = self.projector1(x_residual1)
        x_residual2 = x_residual2.permute(0, 1, 3, 2).contiguous()

        return x_residual2

    def get_l1_regularization(self):
        l1_reg = self.mbda.get_l1_regularization()

        if hasattr(self.graph_conv, 'fusion_gate'):
            device = next(self.graph_conv.fusion_gate.parameters()).device
            l1_reg = l1_reg.to(device)
            for param in self.graph_conv.fusion_gate.parameters():
                l1_reg += torch.sum(torch.abs(param)) * 0.005

        return l1_reg


class AmTGBiM(nn.Module):
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict, len_input,
                 adj_mx, attention_heads=4, mamba_d_state='auto', mamba_d_conv=2, mamba_expand=1,
                 k_hop=2, mask_threshold=0.5, feature_weight=0.5, enhanced_pe=True, fusion_gate_dropout=0.05):
        super().__init__()
        self.Block = AmTGBiMBlock(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, len_input, adj_mx,
                                  attention_heads, mamba_d_state, mamba_d_conv, mamba_expand,
                                  k_hop, mask_threshold, feature_weight, enhanced_pe, fusion_gate_dropout)
        self.DEVICE = DEVICE
        self.projector3 = nn.Linear(len_input, num_for_predict, bias=True)
        self.to(DEVICE)

    def forward(self, x, batch_idx=None, total_batches=None, epoch=None):
        x = self.Block(x, batch_idx, total_batches, epoch)
        output = torch.squeeze(x, dim=2)
        output = self.projector3(output)
        output_final = output.permute(0, 2, 1)
        return output_final

    def get_l1_regularization(self):
        return self.Block.get_l1_regularization()


def make_AmTGBiM(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx, num_for_predict,
                 len_input, attention_heads=4, mamba_d_state='auto', mamba_d_conv=2, mamba_expand=1,
                 k_hop=2, mask_threshold=0.5, feature_weight=0.5, enhanced_pe=True, fusion_gate_dropout=0.05):
    if nb_chev_filter % attention_heads != 0:
        new_nb_chev_filter = ((nb_chev_filter + attention_heads - 1) // attention_heads) * attention_heads
        print(f"Adjusting nb_chev_filter: {nb_chev_filter} -> {new_nb_chev_filter}")
        nb_chev_filter = new_nb_chev_filter

    model = AmTGBiM(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_for_predict,
                    len_input, adj_mx, attention_heads, mamba_d_state, mamba_d_conv, mamba_expand,
                    k_hop, mask_threshold, feature_weight, enhanced_pe, fusion_gate_dropout)

    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model


class Mamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        A = -torch.exp(self.A_log.float())
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,
                None,
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            if conv_state is not None:
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )
        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())
        if selective_state_update is None:
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, mamba, mamba_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.mamba = mamba
        self.mamba_r = mamba_r
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.forward_weight = nn.Parameter(torch.tensor(0.5))
        self.backward_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        forward_output = self.mamba(x)
        backward_output = self.mamba_r(x.flip(dims=[1])).flip(dims=[1])
        new_x = self.forward_weight * forward_output + self.backward_weight * backward_output
        attn = 1
        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns