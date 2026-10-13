import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import pdb
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def eval_anomaly_node_detection(model, data, batch_size, n_neighbors, device,
                                only_rec_score = False, only_drift_score=False, test_inference_time=False):
    pred_score = np.zeros(len(data.sources))
    pred_mask = np.zeros(len(data.sources), dtype=bool)
    timestamps_batch_all = []
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        model.eval()

        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            if test_inference_time:
                for idx in range(s_idx, e_idx):
                    model.neighbor_finder.node_to_neighbors[data.sources[idx]] = np.append(model.neighbor_finder.node_to_neighbors[data.sources[idx]], data.destinations[idx])
                    model.neighbor_finder.node_to_neighbors[data.destinations[idx]] = np.append(model.neighbor_finder.node_to_neighbors[data.destinations[idx]], data.sources[idx])
                    model.neighbor_finder.node_to_edge_timestamps[data.sources[idx]] = np.append(model.neighbor_finder.node_to_edge_timestamps[data.sources[idx]], data.timestamps[idx])
                    model.neighbor_finder.node_to_edge_timestamps[data.destinations[idx]] = np.append(model.neighbor_finder.node_to_edge_timestamps[data.destinations[idx]], data.timestamps[idx])

            sources_batch = torch.from_numpy(data.sources[s_idx: e_idx]).long().to(device)
            destinations_batch = torch.from_numpy(data.destinations[s_idx: e_idx]).long().to(device)
            timestamps_batch = torch.from_numpy(data.timestamps[s_idx:e_idx]).float().to(device)

            timestamps_batch_all.extend(timestamps_batch.cpu().numpy().tolist())

            src_neighbors_batch_np, _, src_neighbors_time_batch_np = model.neighbor_finder.get_temporal_neighbor(data.sources[s_idx: e_idx], data.timestamps[s_idx:e_idx], n_neighbors)
            dst_neighbors_batch_np, _, dst_neighbors_time_batch_np = model.neighbor_finder.get_temporal_neighbor(data.destinations[s_idx: e_idx], data.timestamps[s_idx:e_idx], n_neighbors)
            src_neighbors_batch = torch.from_numpy(src_neighbors_batch_np).long().to(device)
            dst_neighbors_batch = torch.from_numpy(dst_neighbors_batch_np).long().to(device)
            src_neighbors_time_batch = torch.from_numpy(src_neighbors_time_batch_np).long().to(device)
            dst_neighbors_time_batch = torch.from_numpy(dst_neighbors_time_batch_np).long().to(device)


            positive_memory_score, drift_score, _, _ = model.compute_anomaly_score(sources_batch, destinations_batch,
                                                                            timestamps_batch, src_neighbors_batch, dst_neighbors_batch,
                                                                            src_neighbors_time_batch, dst_neighbors_time_batch, n_neighbors)


            if np.isnan(positive_memory_score.reshape(-1).cpu().numpy()).sum()>0:
                pdb.set_trace()

            if only_drift_score: # ablation study
                pred_score[s_idx: e_idx] = (-(drift_score).reshape(-1).cpu().numpy() + 1)/2
            elif only_rec_score: # ablation study
                pred_score[s_idx: e_idx] = (-(positive_memory_score).reshape(-1).cpu().numpy() + 1)/2
            else: #结合了时间和结构两个维度来综合评估异常程度，分数越高表示越可能是异常样本。
                pred_score[s_idx: e_idx] = (-(drift_score).reshape(-1).cpu().numpy()-(positive_memory_score).reshape(-1).cpu().numpy() + 2)/4

        auc_roc = roc_auc_score(data.labels, pred_score)

        return auc_roc, pred_score, _, timestamps_batch_all


def eval_anomaly_node_detection_with_vae(model, test_data, batch_size, n_neighbors, device,
                                         only_rec_score=False, only_drift_score=False,
                                         test_inference_time=False, use_vae=True):
    """
    支持VAE的异常检测评估函数
    """
    from sklearn.metrics import roc_auc_score
    import math
    import torch
    import numpy as np
    from utils.neighor_finder import get_neighbor_finder

    model.eval()

    test_sources = torch.from_numpy(test_data.sources).long().to(device)
    test_destinations = torch.from_numpy(test_data.destinations).long().to(device)
    test_timestamps = torch.from_numpy(test_data.timestamps).float().to(device)
    test_labels = torch.from_numpy(test_data.labels).long().to(device)

    # 获取邻居信息
    if test_inference_time:
        ngh_finder = model.neighbor_finder
    else:
        max_idx = max(test_data.sources.max(), test_data.destinations.max())
        ngh_finder = get_neighbor_finder(test_data, uniform=False, max_node_idx=max_idx)

    src_neighbors, _, src_neighbors_time = ngh_finder.get_temporal_neighbor_tqdm(
        test_data.sources, test_data.timestamps, n_neighbors)
    dst_neighbors, _, dst_neighbors_time = ngh_finder.get_temporal_neighbor_tqdm(
        test_data.destinations, test_data.timestamps, n_neighbors)

    test_src_neighbors = torch.from_numpy(src_neighbors).long().to(device)
    test_dst_neighbors = torch.from_numpy(dst_neighbors).long().to(device)
    test_src_neighbors_time = torch.from_numpy(src_neighbors_time).long().to(device)
    test_dst_neighbors_time = torch.from_numpy(dst_neighbors_time).long().to(device)

    num_test_instance = len(test_data.sources)
    num_test_batch = math.ceil(num_test_instance / batch_size)

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for k in range(num_test_batch):
            s_idx = k * batch_size
            e_idx = min(num_test_instance, s_idx + batch_size)

            sources_batch = test_sources[s_idx:e_idx]
            destinations_batch = test_destinations[s_idx:e_idx]
            timestamps_batch = test_timestamps[s_idx:e_idx]
            src_neighbors_batch = test_src_neighbors[s_idx:e_idx]
            dst_neighbors_batch = test_dst_neighbors[s_idx:e_idx]
            src_neighbors_time_batch = test_src_neighbors_time[s_idx:e_idx]
            dst_neighbors_time_batch = test_dst_neighbors_time[s_idx:e_idx]
            labels_batch = test_labels[s_idx:e_idx]

            # 计算异常分数
            result = model.compute_anomaly_score(
                sources_batch, destinations_batch, timestamps_batch,
                src_neighbors_batch, dst_neighbors_batch,
                src_neighbors_time_batch, dst_neighbors_time_batch, n_neighbors)

            if use_vae and hasattr(model, 'use_vae') and model.use_vae and len(result) == 6:
                # 包含VAE分数的情况
                (pos_recovery_score, pos_drift_score, dst_recovery_score,
                 dst_drift_score, src_vae_score, dst_vae_score) = result

                # 组合异常分数
                if only_rec_score:
                    batch_scores = -(pos_recovery_score + dst_recovery_score) / 2 + \
                                   (src_vae_score + dst_vae_score) / 2
                elif only_drift_score:
                    batch_scores = -(pos_drift_score + dst_drift_score) / 2 + \
                                   (src_vae_score + dst_vae_score) / 2
                else:
                    # 综合所有分数
                    traditional_score = -(pos_recovery_score + dst_recovery_score +
                                          pos_drift_score + dst_drift_score) / 4
                    vae_score = (src_vae_score + dst_vae_score) / 2
                    batch_scores = traditional_score + 0.3 * vae_score
            else:
                # 不使用VAE的情况
                pos_recovery_score, pos_drift_score, dst_recovery_score, dst_drift_score = result[:4]

                if only_rec_score:
                    batch_scores = -(pos_recovery_score + dst_recovery_score) / 2
                elif only_drift_score:
                    batch_scores = -(pos_drift_score + dst_drift_score) / 2
                else:
                    batch_scores = -(pos_recovery_score + dst_recovery_score +
                                     pos_drift_score + dst_drift_score) / 4

            all_scores.extend(batch_scores.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    # 计算AUC
    auc = roc_auc_score(all_labels, all_scores)

    return auc, np.array(all_scores), np.array(all_labels), test_timestamps.cpu().numpy()