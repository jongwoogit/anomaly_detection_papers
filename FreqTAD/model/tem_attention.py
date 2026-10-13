import torch
from torch import nn

from utils.utils import MergeLayer
import pdb


# class TemporalAttentionLayer_recovery(torch.nn.Module):
#     """
#     Temporal attention layer. Return the temporal embedding of a node given the node itself,
#     its neighbors and the edge timestamps.
#     """
#
#     def __init__(self, n_node_features, n_neighbors_features, time_dim,
#                  output_dimension, n_head=2,
#                  dropout=0.1):
#         super(TemporalAttentionLayer_recovery, self).__init__()
#
#         self.n_head = n_head
#
#         self.feat_dim = n_node_features
#         self.time_dim = time_dim
#
#         self.query_dim = time_dim
#         self.key_dim = n_neighbors_features + time_dim
#         self.val_dim = n_neighbors_features + time_dim
#
#         self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
#                                                        kdim=self.key_dim,
#                                                        vdim=self.val_dim,
#                                                        num_heads=n_head,
#                                                        dropout=dropout)
#
#     def forward(self, src_time_features, neighbors_features,
#                 neighbors_time_features, neighbors_padding_mask):
#         """
#         "Temporal attention model
#         :param src_node_features: float Tensor of shape [batch_size, n_node_features]
#         :param src_time_features: float Tensor of shape [batch_size, 1, time_dim]
#         :param neighbors_features: float Tensor of shape [batch_size, n_neighbors, n_node_features]
#         :param neighbors_time_features: float Tensor of shape [batch_size, n_neighbors,
#         time_dim]
#         :param edge_features: float Tensor of shape [batch_size, n_neighbors, n_edge_features]
#         :param neighbors_padding_mask: float Tensor of shape [batch_size, n_neighbors]
#         :return:
#         attn_output: float Tensor of shape [1, batch_size, n_node_features]
#         attn_output_weights: [batch_size, 1, n_neighbors]
#         """
#
#         query = src_time_features
#         key = torch.cat([neighbors_features, neighbors_time_features], dim=2)
#         val = torch.cat([neighbors_features, neighbors_time_features], dim=2)
#
#         # Reshape tensors so to expected shape by multi head attention
#         query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
#         key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]
#         val = val.permute([1, 0, 2])
#
#         # Compute mask of which source nodes have no valid neighbors
#         invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
#         # If a source node has no valid neighbor, set it's first neighbor to be valid. This will
#         # force the attention to just 'attend' on this neighbor (which has the same features as all
#         # the others since they are fake neighbors) and will produce an equivalent result to the
#         # original tgat paper which was forcing fake neighbors to all have same attention of 1e-10
#         neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False
#
#         attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=val,
#                                                                   key_padding_mask=neighbors_padding_mask)
#
#         attn_output = attn_output.squeeze()
#         attn_output_weights = attn_output_weights.squeeze()
#
#         # Source nodes with no neighbors have an all zero attention output. The attention output is
#         # then added or concatenated to the original source node features and then fed into an MLP.
#         # This means that an all zero vector is not used.
#         attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
#         attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)
#
#         # Skip connection with temporal attention over neighborhood and the features of the node itself
#
#         return attn_output, attn_output_weights


# class TemporalAttentionLayer_recovery(torch.nn.Module):
#     """
#     频域时序注意力层，在频域中计算注意力权重
#     """
#     def __init__(self, n_node_features, n_neighbors_features, time_dim,
#                  output_dimension, n_head=2, dropout=0.1):
#         super().__init__()
#
#         self.n_head = n_head
#         self.time_dim = time_dim
#         self.n_neighbors_features = n_neighbors_features
#         self.output_dimension = output_dimension
#
#         self.kv_dim = n_neighbors_features + time_dim  # 512
#
#         self.freq_transform = nn.Sequential(
#             nn.Linear(time_dim, time_dim),
#             nn.ReLU(),
#             nn.Linear(time_dim, time_dim)
#         )
#         self.query_projection = nn.Linear(time_dim, self.kv_dim)
#
#         self.freq_attention = nn.MultiheadAttention(
#             embed_dim=self.kv_dim,  # 修改为kv_dim
#             num_heads=n_head,
#             dropout=dropout,
#             batch_first=False
#         )
#
#         # 时域注意力 - 现在所有维度都是kv_dim
#         self.time_attention = nn.MultiheadAttention(
#             embed_dim=self.kv_dim,  # 修改为kv_dim
#             num_heads=n_head,
#             dropout=dropout,
#             batch_first=False
#         )
#
#         # 时频融合 - 现在输入维度是正确的
#         fusion_input_dim = 2 * self.kv_dim  # 2 * 512 = 1024
#         self.time_freq_fusion = nn.Sequential(
#             nn.Linear(fusion_input_dim, fusion_input_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(fusion_input_dim // 2, output_dimension)
#         )
#
#     def forward(self, src_time_features, neighbors_features,
#                 neighbors_time_features, neighbors_padding_mask):
#
#         try:
#             batch_size = src_time_features.shape[0]
#
#             # 频域变换
#             freq_src_time = self._to_frequency_domain(src_time_features)
#             freq_neighbor_time = self._to_frequency_domain(neighbors_time_features)
#
#             # 处理无效邻居掩码
#             invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
#             neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False
#
#             # 频域注意力
#             freq_query_projected = self.query_projection(freq_src_time)  # [200, 1, 512]
#             freq_query = freq_query_projected.permute([1, 0, 2])  # [1, 200, 512]
#             freq_key = torch.cat([neighbors_features, freq_neighbor_time], dim=2).permute([1, 0, 2])  # [20, 200, 512]
#             freq_val = torch.cat([neighbors_features, freq_neighbor_time], dim=2).permute([1, 0, 2])  # [20, 200, 512]
#
#             freq_output, _ = self.freq_attention(
#                 query=freq_query, key=freq_key, value=freq_val,
#                 key_padding_mask=neighbors_padding_mask
#             )
#
#             # 时域注意力
#             time_query_projected = self.query_projection(src_time_features)  # [200, 1, 512]
#             time_query = time_query_projected.permute([1, 0, 2])  # [1, 200, 512]
#             time_key = torch.cat([neighbors_features, neighbors_time_features], dim=2).permute(
#                 [1, 0, 2])  # [20, 200, 512]
#             time_val = torch.cat([neighbors_features, neighbors_time_features], dim=2).permute(
#                 [1, 0, 2])  # [20, 200, 512]
#
#             time_output, _ = self.time_attention(
#                 query=time_query, key=time_key, value=time_val,
#                 key_padding_mask=neighbors_padding_mask
#             )
#
#             # 处理输出维度
#             freq_output = freq_output.squeeze(0)  # [200, 512]
#             time_output = time_output.squeeze(0)  # [200, 512]
#
#             # 处理无效邻居
#             freq_output = freq_output.masked_fill(invalid_neighborhood_mask, 0)
#             time_output = time_output.masked_fill(invalid_neighborhood_mask, 0)
#
#             # time_weight = 1.6
#             # freq_weight = 0.4
#
#             # 对每个张量加权后再拼接
#             # weighted_time = time_weight * time_output
#             # weighted_freq = freq_weight * freq_output
#             # 融合时域和频域信息
#             combined_output = torch.cat([time_output, freq_output], dim=-1)
#             # combined_output = torch.cat([weighted_time, weighted_freq], dim=-1)  # [200, 1024]
#
#             final_output = self.time_freq_fusion(combined_output)
#             return final_output, None
#
#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             raise e
#
#     def _to_frequency_domain(self, time_features):
#         """时域到频域变换"""
#         return self.freq_transform(time_features)


import torch
from torch import nn
import torch.nn.functional as F


class TemporalAttentionLayer_recovery(torch.nn.Module):
    """
    基于FFT的时频域融合注意力层
    """

    def __init__(self, n_node_features, n_neighbors_features, time_dim,
                 output_dimension, n_head=2, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.time_dim = time_dim
        self.n_neighbors_features = n_neighbors_features
        self.output_dimension = output_dimension

        self.kv_dim = n_neighbors_features + time_dim  # 512

        # FFT频域处理
        self.freq_dim = time_dim * 2  # 实部+虚部

        # 频域特征投影
        self.freq_projection = nn.Sequential(
            nn.Linear(self.freq_dim, time_dim),
            nn.LayerNorm(time_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 查询投影
        self.query_projection = nn.Linear(time_dim, self.kv_dim)

        # 频域注意力
        self.freq_attention = nn.MultiheadAttention(
            embed_dim=self.kv_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=False
        )

        # 时域注意力
        self.time_attention = nn.MultiheadAttention(
            embed_dim=self.kv_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=False
        )

        # 时频融合网络
        fusion_input_dim = 2 * self.kv_dim  # 时域 + 频域
        self.time_freq_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.LayerNorm(fusion_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_dim // 2, fusion_input_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_input_dim // 4, output_dimension)
        )

        # 频域权重学习
        self.freq_gate = nn.Sequential(
            nn.Linear(self.kv_dim, self.kv_dim // 4),
            nn.ReLU(),
            nn.Linear(self.kv_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, src_time_features, neighbors_features,
                neighbors_time_features, neighbors_padding_mask):
        try:
            batch_size = src_time_features.shape[0]

            # FFT频域变换
            freq_src_time = self._fft_transform(src_time_features)
            freq_neighbor_time = self._fft_transform(neighbors_time_features)

            # 处理无效邻居掩码
            invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
            neighbors_padding_mask_fixed = neighbors_padding_mask.clone()
            neighbors_padding_mask_fixed[invalid_neighborhood_mask.squeeze(), 0] = False

            # === 频域注意力 ===
            freq_query_projected = self.query_projection(freq_src_time)  # [batch, 1, kv_dim]
            freq_query = freq_query_projected.permute([1, 0, 2])  # [1, batch, kv_dim]

            # 构建频域的key和value
            freq_key = torch.cat([neighbors_features, freq_neighbor_time], dim=2)  # [batch, n_neighbors, kv_dim]
            freq_val = freq_key.clone()

            freq_key = freq_key.permute([1, 0, 2])  # [n_neighbors, batch, kv_dim]
            freq_val = freq_val.permute([1, 0, 2])  # [n_neighbors, batch, kv_dim]

            freq_output, freq_weights = self.freq_attention(
                query=freq_query,
                key=freq_key,
                value=freq_val,
                key_padding_mask=neighbors_padding_mask_fixed
            )

            # === 时域注意力 ===
            time_query_projected = self.query_projection(src_time_features)  # [batch, 1, kv_dim]
            time_query = time_query_projected.permute([1, 0, 2])  # [1, batch, kv_dim]

            time_key = torch.cat([neighbors_features, neighbors_time_features], dim=2)  # [batch, n_neighbors, kv_dim]
            time_val = time_key.clone()

            time_key = time_key.permute([1, 0, 2])  # [n_neighbors, batch, kv_dim]
            time_val = time_val.permute([1, 0, 2])  # [n_neighbors, batch, kv_dim]

            time_output, time_weights = self.time_attention(
                query=time_query,
                key=time_key,
                value=time_val,
                key_padding_mask=neighbors_padding_mask_fixed
            )

            # 处理输出维度
            freq_output = freq_output.squeeze(0)  # [batch, kv_dim]
            time_output = time_output.squeeze(0)  # [batch, kv_dim]

            # 处理无效邻居
            freq_output = freq_output.masked_fill(invalid_neighborhood_mask, 0)
            time_output = time_output.masked_fill(invalid_neighborhood_mask, 0)

            # === 自适应时频融合 ===
            # 计算频域信息的重要性权重
            freq_importance = 0.2#self.freq_gate(freq_output)  # [batch, 1]

            # 加权融合
            weighted_freq = freq_output * freq_importance
            weighted_time = time_output * (1 - freq_importance)

            # 最终融合
            combined_output = torch.cat([weighted_time, weighted_freq], dim=-1)  # [batch, 2*kv_dim]
            final_output = self.time_freq_fusion(combined_output)  # [batch, output_dim]

            return final_output, {
                'time_weights': time_weights,
                'freq_weights': freq_weights,
                'freq_importance': freq_importance
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    def _fft_transform(self, time_features):
        """
        Args:
            time_features: [batch_size, seq_len, time_dim] 或 [batch_size, n_neighbors, time_dim]
        Returns:
            频域特征: [batch_size, seq_len, time_dim] 或 [batch_size, n_neighbors, time_dim]
        """

        # 对最后一个维度进行FFT
        fft_result = torch.fft.fft(time_features, dim=-1)
        magnitude, phase, low_freq, high_freq = self._extract_frequency_components(fft_result)
        # 提取实部和虚部
        real_part = fft_result.real  # [batch, seq, time_dim]
        imag_part = fft_result.imag  # [batch, seq, time_dim]
        # 拼接实部和虚部
        freq_features = torch.cat([real_part, imag_part], dim=-1)  # [batch, seq, 2*time_dim]
        # 投影回原始维度
        freq_features = self.freq_projection(freq_features)  # [batch, seq, time_dim]

        # freq_components = {
        #     'magnitude': magnitude,
        #     'phase': phase,
        #     'low_freq': low_freq,
        #     'high_freq': high_freq
        # }

        return freq_features

    def _extract_frequency_components(self, fft_result):
        """
        提取频域的关键成分
        """
        # 计算幅度谱
        magnitude = torch.abs(fft_result)
        # 计算相位谱
        phase = torch.angle(fft_result)
        # 提取低频成分（前1/4）和高频成分（后1/4）
        freq_dim = fft_result.shape[-1]
        low_freq = magnitude[..., :freq_dim // 4]
        high_freq = magnitude[..., -freq_dim // 4:]

        return magnitude, phase, low_freq, high_freq

