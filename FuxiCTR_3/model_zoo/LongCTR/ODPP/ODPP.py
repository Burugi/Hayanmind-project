# =========================================================================
# Copyright (C) 2025. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.utils import not_in_whitelist


class SegmenterModule(nn.Module):
    """Segments the sequence into fixed-length segments."""

    def __init__(self, seg_len):
        super(SegmenterModule, self).__init__()
        self.seg_len = seg_len

    def forward(self, sequence_emb, mask=None):
        """
        Args:
            sequence_emb: (B, L, D)
            mask: (B, L)
        Returns:
            segmented_emb: (B, T, seg_len, D)
            segmented_mask: (B, T, seg_len) if mask is provided
        """
        B, L, D = sequence_emb.shape
        T = L // self.seg_len

        # Truncate to make divisible by seg_len
        truncated_len = T * self.seg_len
        sequence_emb = sequence_emb[:, :truncated_len, :]

        # Reshape to segments
        segmented_emb = sequence_emb.view(B, T, self.seg_len, D)

        segmented_mask = None
        if mask is not None:
            mask = mask[:, :truncated_len]
            segmented_mask = mask.view(B, T, self.seg_len)

        return segmented_emb, segmented_mask


class DistributionalOrderedEncoder(nn.Module):
    """Encodes segment distribution with position-aware and target-aware attention."""

    def __init__(self, input_dim, hidden_dim, output_dim, use_target_attention=True):
        super(DistributionalOrderedEncoder, self).__init__()
        self.use_target_attention = use_target_attention

        self.position_mlp = MLP_Block(
            input_dim=1,
            hidden_units=[hidden_dim],
            hidden_activations="ReLU",
            output_dim=hidden_dim,
            batch_norm=False
        )

        # Attention with optional target conditioning
        attention_input_dim = input_dim + hidden_dim
        if use_target_attention:
            attention_input_dim += input_dim  # Add target dimension

        self.attention_mlp = MLP_Block(
            input_dim=attention_input_dim,
            hidden_units=[hidden_dim],
            hidden_activations="ReLU",
            output_dim=1,
            batch_norm=False
        )

        self.output_proj = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, segmented_emb, segmented_mask=None, target_emb=None):
        """
        Args:
            segmented_emb: (B, T, seg_len, D_embed)
            segmented_mask: (B, T, seg_len)
            target_emb: (B, D_embed) - target item embedding for attention
        Returns:
            segment_repr: (B, T, D_dist)
        """
        B, T, seg_len, D = segmented_emb.shape

        # Compute relative positions (0 to 1)
        positions = torch.arange(seg_len, dtype=torch.float32, device=segmented_emb.device)
        positions = positions / max(seg_len - 1, 1)
        positions = positions.view(1, 1, seg_len, 1).expand(B, T, -1, -1)

        # Position encoding
        pos_encoding = self.position_mlp(positions)  # (B, T, seg_len, hidden_dim)

        # Combine event embedding with position encoding
        combined = torch.cat([segmented_emb, pos_encoding], dim=-1)

        # Add target conditioning if enabled
        if self.use_target_attention and target_emb is not None:
            target_expanded = target_emb.unsqueeze(1).unsqueeze(2).expand(-1, T, seg_len, -1)
            combined = torch.cat([combined, target_expanded], dim=-1)

        # Compute attention scores
        attention_scores = self.attention_mlp(combined).squeeze(-1)  # (B, T, seg_len)

        # Apply mask if provided
        if segmented_mask is not None:
            attention_scores = attention_scores.masked_fill(segmented_mask == 0, -1e9)

        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, T, seg_len)

        # Weighted sum
        segment_repr = torch.einsum('bts,btsd->btd', attention_weights, segmented_emb)

        # Project to output dimension and normalize
        segment_repr = self.output_proj(segment_repr)
        segment_repr = self.layer_norm(segment_repr)

        return segment_repr


class PositionAwareTopKPooling(nn.Module):
    """Selects top-K important events with position and target awareness."""

    def __init__(self, input_dim, max_seq_len, top_k, hidden_dim, output_dim, use_target_attention=True):
        super(PositionAwareTopKPooling, self).__init__()
        self.top_k = top_k
        self.input_dim = input_dim
        self.use_target_attention = use_target_attention

        self.max_position_embeddings = 10000
        self.position_embedding = nn.Embedding(self.max_position_embeddings, input_dim)

        # Importance scoring with optional target conditioning
        importance_input_dim = input_dim * 2
        if use_target_attention:
            importance_input_dim += input_dim

        self.importance_mlp = MLP_Block(
            input_dim=importance_input_dim,
            hidden_units=[hidden_dim],
            hidden_activations="ReLU",
            output_dim=1,
            batch_norm=False
        )

        self.encoding_mlp = MLP_Block(
            input_dim=input_dim * 2,
            hidden_units=[hidden_dim],
            hidden_activations="ReLU",
            output_dim=output_dim,
            batch_norm=False
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, sequence_emb, mask=None, target_emb=None):
        """
        Args:
            sequence_emb: (B, L, D_embed)
            mask: (B, L)
            target_emb: (B, D_embed) - target item embedding for importance scoring
        Returns:
            pooled_repr: (B, D_pos)
        """
        B, L, D = sequence_emb.shape

        # Get position embeddings
        positions = torch.arange(L, device=sequence_emb.device).unsqueeze(0).expand(B, -1)
        positions = torch.clamp(positions, max=self.max_position_embeddings - 1)
        pos_emb = self.position_embedding(positions)

        # Combine event and position embeddings
        combined_emb = torch.cat([sequence_emb, pos_emb], dim=-1)

        # Add target conditioning for importance scoring
        if self.use_target_attention and target_emb is not None:
            target_expanded = target_emb.unsqueeze(1).expand(-1, L, -1)
            importance_input = torch.cat([combined_emb, target_expanded], dim=-1)
        else:
            importance_input = combined_emb

        # Compute importance scores
        importance_scores = self.importance_mlp(importance_input).squeeze(-1)

        # Apply mask if provided
        if mask is not None:
            importance_scores = importance_scores.masked_fill(mask == 0, -1e9)

        # Select top-K indices
        top_k = min(self.top_k, L)
        topk_values, topk_indices = torch.topk(importance_scores, k=top_k, dim=-1)

        # Sort indices to preserve order
        topk_indices_sorted, _ = torch.sort(topk_indices, dim=-1)

        # Gather top-K embeddings
        batch_indices = torch.arange(B, device=sequence_emb.device).unsqueeze(1).expand(-1, top_k)
        topk_combined_emb = combined_emb[batch_indices, topk_indices_sorted]

        # Encode and pool with attention-weighted aggregation
        encoded = self.encoding_mlp(topk_combined_emb)

        # Use softmax attention for weighted pooling
        topk_scores = topk_values.gather(1, torch.argsort(topk_indices, dim=-1))
        attn_weights = F.softmax(topk_scores, dim=-1)
        pooled_repr = torch.einsum('bk,bkd->bd', attn_weights, encoded)

        pooled_repr = self.layer_norm(pooled_repr)

        return pooled_repr


class FusionModule(nn.Module):
    """Fuses distributional and position-aware representations with gating."""

    def __init__(self, dist_dim, pos_dim, output_dim, use_distribution_encoder=True, use_position_pooling=True):
        super(FusionModule, self).__init__()
        self.use_distribution_encoder = use_distribution_encoder
        self.use_position_pooling = use_position_pooling

        # Calculate input dimension based on which components are used
        fusion_input_dim = 0
        if use_distribution_encoder:
            fusion_input_dim += dist_dim
        if use_position_pooling:
            fusion_input_dim += pos_dim

        self.fusion_layer = nn.Linear(fusion_input_dim, output_dim)

        # Gating mechanism for adaptive fusion
        if use_distribution_encoder and use_position_pooling:
            self.gate_layer = nn.Sequential(
                nn.Linear(dist_dim + pos_dim, output_dim),
                nn.Sigmoid()
            )
            self.dist_proj = nn.Linear(dist_dim, output_dim)
            self.pos_proj = nn.Linear(pos_dim, output_dim)

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, z_dist=None, z_pos=None):
        """
        Args:
            z_dist: (B, T, D_dist) or (B, D_dist) after pooling
            z_pos: (B, D_pos)
        Returns:
            fused_repr: (B, D_fused)
        """
        representations = []

        if self.use_distribution_encoder and z_dist is not None:
            if z_dist.dim() == 3:
                z_dist = z_dist.mean(dim=1)
            representations.append(z_dist)

        if self.use_position_pooling and z_pos is not None:
            representations.append(z_pos)

        # Gated fusion when both components are available
        if len(representations) == 2 and self.use_distribution_encoder and self.use_position_pooling:
            combined = torch.cat(representations, dim=-1)
            gate = self.gate_layer(combined)
            dist_transformed = self.dist_proj(representations[0])
            pos_transformed = self.pos_proj(representations[1])
            fused_repr = gate * dist_transformed + (1 - gate) * pos_transformed
        else:
            fused = torch.cat(representations, dim=-1)
            fused_repr = self.fusion_layer(fused)

        fused_repr = self.layer_norm(fused_repr)

        return fused_repr


class ODPP(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="ODPP",
                 gpu=-1,
                 embedding_dim=64,
                 seg_len=30,
                 top_k=20,
                 dist_hidden_dim=64,
                 dist_output_dim=128,
                 pos_hidden_dim=64,
                 pos_output_dim=128,
                 fusion_output_dim=256,
                 dnn_hidden_units=[512, 256, 128],
                 dnn_activations="ReLU",
                 use_distribution_encoder=True,
                 use_position_pooling=True,
                 use_target_attention=True,
                 learning_rate=1e-3,
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(ODPP, self).__init__(feature_map,
                                   model_id=model_id,
                                   gpu=gpu,
                                   embedding_regularizer=embedding_regularizer,
                                   net_regularizer=net_regularizer,
                                   **kwargs)

        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.seg_len = seg_len
        self.top_k = top_k
        self.accumulation_steps = accumulation_steps
        self.use_distribution_encoder = use_distribution_encoder
        self.use_position_pooling = use_position_pooling
        self.use_target_attention = use_target_attention

        # Calculate item embedding dimension and batch embedding dimension
        self.item_info_dim = 0
        self.batch_emb_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim)
            elif spec.get("type") not in ["meta"]:
                # Non-meta, non-item features go into batch_dict
                self.batch_emb_dim += spec.get("embedding_dim", embedding_dim)

        # Embedding layer
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        # Modules
        if use_distribution_encoder:
            self.segmenter = SegmenterModule(seg_len)
            self.dist_encoder = DistributionalOrderedEncoder(
                input_dim=self.item_info_dim,
                hidden_dim=dist_hidden_dim,
                output_dim=dist_output_dim,
                use_target_attention=use_target_attention
            )

        if use_position_pooling:
            self.pos_pooling = PositionAwareTopKPooling(
                input_dim=self.item_info_dim,
                max_seq_len=kwargs.get("max_len", 1000),
                top_k=top_k,
                hidden_dim=pos_hidden_dim,
                output_dim=pos_output_dim,
                use_target_attention=use_target_attention
            )

        self.fusion = FusionModule(
            dist_dim=dist_output_dim,
            pos_dim=pos_output_dim,
            output_dim=fusion_output_dim,
            use_distribution_encoder=use_distribution_encoder,
            use_position_pooling=use_position_pooling
        )

        # Calculate DNN input dimension: batch_emb + fused_repr + target_emb
        dnn_input_dim = self.batch_emb_dim + fusion_output_dim + self.item_info_dim

        # Backbone MLP
        self.dnn = MLP_Block(
            input_dim=dnn_input_dim,
            output_dim=1,
            hidden_units=dnn_hidden_units,
            hidden_activations=dnn_activations,
            output_activation=self.output_activation,
            dropout_rates=net_dropout,
            batch_norm=batch_norm
        )

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        emb_list = []

        # Batch features (non-sequential)
        if batch_dict:
            batch_emb = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(batch_emb)

        # Item features (sequential) - includes target item at the end
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)

        # Separate target item and history sequence (like SDIM)
        target_emb = item_feat_emb[:, -1, :]  # (B, D) - target item
        sequence_emb = item_feat_emb[:, :-1, :]  # (B, L, D) - history sequence
        sequence_mask = mask  # (B, L) - mask matches history sequence length

        # Process through ODPP modules
        z_dist = None
        z_pos = None

        if self.use_distribution_encoder:
            # Segmentation
            segmented_emb, segmented_mask = self.segmenter(sequence_emb, sequence_mask)
            # Distributional encoding with target attention
            z_dist = self.dist_encoder(segmented_emb, segmented_mask, target_emb)

        if self.use_position_pooling:
            # Position-aware top-K pooling with target attention
            z_pos = self.pos_pooling(sequence_emb, sequence_mask, target_emb)

        # Fusion
        fused_repr = self.fusion(z_dist, z_pos)
        emb_list.append(fused_repr)

        # Add target item embedding for direct feature interaction
        emb_list.append(target_emb)

        # Concatenate all features
        feature_emb = torch.cat(emb_list, dim=-1)

        # Final prediction
        y_pred = self.dnn(feature_emb)

        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_inputs(self, inputs, feature_source=None):
        batch_dict, item_dict, mask = inputs
        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
        return X_dict, item_dict, mask.to(self.device)

    def get_labels(self, inputs):
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)

    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def train_step(self, batch_data):
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss
