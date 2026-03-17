import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.utils import not_in_whitelist
import os
import csv
import numpy as np


class MultiScaleSubsequenceExtractor(nn.Module):
    """Extract subsequences using sliding windows of multiple sizes and mean-pool each."""

    def __init__(self, window_sizes):
        super().__init__()
        self.window_sizes = window_sizes

    def forward(self, sequence_emb, mask):
        """
        Args:
            sequence_emb: (B, L, D) - item embedding sequence
            mask: (B, L) - 1 for valid positions, 0 for padding
        Returns:
            subseq_vectors: (B, total_windows, D)
        """
        B, L, D = sequence_emb.shape
        all_subseqs = []

        for w in self.window_sizes:
            if L < w:
                continue
            # Unfold to get sliding windows: (B, num_windows, w, D)
            windows = sequence_emb.unfold(1, w, 1)  # (B, num_windows, D, w)
            windows = windows.permute(0, 1, 3, 2)   # (B, num_windows, w, D)
            num_windows = windows.shape[1]

            # Build mask for each window
            mask_windows = mask.unfold(1, w, 1)  # (B, num_windows, w)

            # Mean pooling with mask
            mask_expanded = mask_windows.unsqueeze(-1).float()  # (B, num_windows, w, 1)
            masked_windows = windows * mask_expanded
            window_lens = mask_expanded.sum(dim=2).clamp(min=1)  # (B, num_windows, 1)
            pooled = masked_windows.sum(dim=2) / window_lens  # (B, num_windows, D)

            all_subseqs.append(pooled)

        if not all_subseqs:
            # Fallback: if sequence is too short for any window, return the mean of the sequence
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (sequence_emb * mask_expanded).sum(dim=1, keepdim=True) / mask_expanded.sum(dim=1, keepdim=True).clamp(min=1)
            return pooled

        return torch.cat(all_subseqs, dim=1)  # (B, total_windows, D)


class LearnableCodebook(nn.Module):
    """Learnable prototype codebook with Gumbel-softmax assignment."""

    def __init__(self, num_prototypes, embedding_dim, temperature=1.0):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        # Initialize prototypes with xavier uniform
        proto_data = torch.empty(num_prototypes, embedding_dim)
        nn.init.xavier_uniform_(proto_data.unsqueeze(0))
        self.prototypes = nn.Parameter(proto_data.squeeze(0))

    def forward(self, subseq_vectors):
        """
        Args:
            subseq_vectors: (B, N, D) - subsequence vectors
        Returns:
            cluster_centers: (B, N, D) - assigned prototype vectors
            assignments: (B, N) - hard cluster assignment indices
        """
        # Normalize for cosine similarity
        subseq_norm = F.normalize(subseq_vectors, dim=-1)  # (B, N, D)
        proto_norm = F.normalize(self.prototypes, dim=-1)   # (K, D)

        # Cosine similarity: (B, N, K)
        similarity = torch.matmul(subseq_norm, proto_norm.t())

        if self.training:
            # Gumbel-softmax for differentiable assignment
            soft_assign = F.gumbel_softmax(similarity / self.temperature, tau=1.0, hard=False, dim=-1)
        else:
            # Hard assignment at inference
            soft_assign = torch.zeros_like(similarity)
            indices = similarity.argmax(dim=-1)
            soft_assign.scatter_(-1, indices.unsqueeze(-1), 1.0)

        # Get assigned cluster centers: (B, N, D)
        cluster_centers = torch.matmul(soft_assign, self.prototypes)
        assignments = similarity.argmax(dim=-1)  # (B, N) - hard assignments for logging

        return cluster_centers, assignments


class JourneyMap(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="JourneyMap",
                 gpu=-1,
                 embedding_dim=10,
                 window_sizes=[3, 5, 7],
                 num_prototypes=64,
                 gru_hidden_size=32,
                 gru_num_layers=1,
                 temperature=1.0,
                 net_dropout=0,
                 learning_rate=1e-3,
                 accumulation_steps=1,
                 save_prediction_details=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(JourneyMap, self).__init__(feature_map,
                                         model_id=model_id,
                                         gpu=gpu,
                                         embedding_regularizer=embedding_regularizer,
                                         net_regularizer=net_regularizer,
                                         **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.accumulation_steps = accumulation_steps
        self.save_prediction_details = save_prediction_details

        # Compute item embedding dimension
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim)

        # Embedding layer
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        # Multi-scale subsequence extractor
        self.subsequence_extractor = MultiScaleSubsequenceExtractor(window_sizes)

        # Learnable codebook
        self.codebook = LearnableCodebook(num_prototypes, self.item_info_dim, temperature)

        # GRU for journey modeling
        self.gru = nn.GRU(
            input_size=self.item_info_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=net_dropout if gru_num_layers > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(net_dropout) if net_dropout > 0 else nn.Identity()

        # Final prediction layer
        self.output_layer = nn.Linear(gru_hidden_size + self.item_info_dim, 1)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)

        # Get item embeddings
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)

        # Split target (last) and sequence (rest)
        target_emb = item_feat_emb[:, -1, :]          # (B, D)
        sequence_emb = item_feat_emb[:, :-1, :]        # (B, L, D)
        seq_mask = mask                                 # (B, L) - mask already excludes target

        # 1. Multi-scale subsequence extraction
        subseq_vectors = self.subsequence_extractor(sequence_emb, seq_mask)  # (B, N_windows, D)

        # 2. Codebook assignment
        cluster_centers, assignments = self.codebook(subseq_vectors)  # (B, N_windows, D), (B, N_windows)

        # 3. GRU over cluster center journey
        gru_out, _ = self.gru(cluster_centers)  # (B, N_windows, hidden_size)
        journey_repr = gru_out[:, -1, :]        # (B, hidden_size) - last hidden state

        # 4. Apply dropout and predict
        journey_repr = self.dropout(journey_repr)
        combined = torch.cat([journey_repr, target_emb], dim=-1)  # (B, hidden_size + D)
        y_pred = self.output_layer(combined)     # (B, 1)

        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)

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

    def save_codebook(self, save_dir):
        """Save codebook prototypes for analysis."""
        os.makedirs(save_dir, exist_ok=True)
        prototypes = self.codebook.prototypes.detach().cpu().numpy()
        np.save(os.path.join(save_dir, "codebook_prototypes.npy"), prototypes)

    def save_predictions_csv(self, save_path, assignments_list, predictions_list, labels_list):
        """Save prediction details for interpretability analysis."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cluster_sequence', 'prediction', 'label'])
            for assignments, pred, label in zip(assignments_list, predictions_list, labels_list):
                cluster_seq = ','.join(map(str, assignments))
                writer.writerow([cluster_seq, f"{pred:.6f}", f"{label:.0f}"])
