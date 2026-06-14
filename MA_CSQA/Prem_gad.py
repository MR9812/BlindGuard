import argparse
import os
from tqdm import tqdm
from data import AgentGraphDataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GATConv, global_mean_pool
from datetime import datetime
from Dominant import GCNModelAE
from TAM import TAMModel
import torch.nn.functional as F
import numpy as np


class PREMDiscriminator(nn.Module):
    """
    PREM判别器，用于计算节点特征与邻居特征的相似度
    """
    
    def __init__(self, n_in: int, n_hidden: int):
        """
        初始化判别器
        
        Args:
            n_in: 输入特征维度
            n_hidden: 隐藏层维度
        """
        super(PREMDiscriminator, self).__init__()
        self.fc_g = nn.Linear(n_in, n_hidden)
        self.fc_n = nn.Linear(n_in, n_hidden)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, features: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 节点特征 [N, D]
            summary: 邻居聚合特征 [N, D]
            
        Returns:
            相似度分数 [1, N]
        """
        # 计算余弦相似度
        s = F.cosine_similarity(self.fc_n(features), self.fc_g(summary))
        return -1 * s.unsqueeze(0)


class PREMModel(nn.Module):
    """
    PREM主模型，用于图异常检测
    """
    
    def __init__(self, n_in: int, n_hidden: int, k: int = 2):
        """
        初始化PREM模型
        
        Args:
            n_in: 输入特征维度
            n_hidden: 隐藏层维度
            k: 聚合步数
        """
        super(PREMModel, self).__init__()
        self.k = k
        self.discriminator = PREMDiscriminator(n_in, n_hidden)
        
        # 缓存机制
        self.cached_weight = None
        self.cached_features_weighted = None
        self.cached_eg = None
        self.cached_en = None
    
    def _aggregate_neighbors(self, features: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        聚合k跳邻居特征
        
        Args:
            features: 节点特征 [N, D]
            edge_index: 边索引 [2, E]
            num_nodes: 节点数量
            
        Returns:
            聚合后的特征 [N, D]
        """
        x = features.clone()
        
        for _ in range(self.k):
            # 计算度数的归一化
            deg = torch.bincount(edge_index[0], minlength=num_nodes).float().clamp(min=1)
            norm = torch.pow(deg, -0.5)
            
            # 对称归一化
            x = x * norm.unsqueeze(1)
            
            # 消息传递
            out = scatter_mean(x[edge_index[1]], edge_index[0], dim=0, dim_size=num_nodes)
            
            # 再次归一化
            x = out * norm.unsqueeze(1)
        
        return x
    
    def _get_diagonal_weight(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """计算对角权重矩阵"""
        # 创建单位矩阵
        identity = torch.eye(num_nodes, device=edge_index.device)
        
        # 聚合单位矩阵
        aggregated = self._aggregate_neighbors(identity, edge_index, num_nodes)
        
        # 提取对角线元素
        return torch.diag(aggregated)
    
    def _preprocess_graph(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int):
        """预处理图数据，计算聚合特征"""
        # 计算对角权重
        weight = self._get_diagonal_weight(edge_index, num_nodes)
        
        # 聚合邻居特征
        aggregated = self._aggregate_neighbors(x, edge_index, num_nodes)
        
        # 计算加权特征和残差
        features_weighted = (x.t() * weight).t()
        eg = (aggregated - features_weighted)
        
        return weight, features_weighted, eg
    
    def _get_prem_data(self, x: torch.Tensor, edge_index: torch.Tensor, device: torch.device):
        """
        为PREM方法准备数据
        
        Args:
            x: 节点特征
            edge_index: 边索引
            device: 设备
            
        Returns:
            en_p, en_n, eg_p, eg_aug: PREM所需的数据
        """
        # 原始特征
        en_p = x
        # eg_p = x  # 简化处理，使用原始特征作为聚合特征
        _, _, eg_p = self._preprocess_graph(x, edge_index, x.shape[0])
        
        # 随机打乱
        perm = torch.randperm(en_p.shape[0])
        en_n = en_p[perm]
        eg_aug = eg_p[perm]
        
        return en_p, en_n, eg_p, eg_aug
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [N, D]
            edge_index: 边索引 [2, E]
            
        Returns:
            异常分数 [1, N]
        """
        num_nodes = x.size(0)
        
        # 检查是否需要重新计算缓存
        if (self.cached_weight is None or 
            self.cached_weight.size(0) != num_nodes or
            self.cached_en is None or
            not torch.equal(self.cached_en, x)):
            
            # 重新计算
            weight, features_weighted, eg = self._preprocess_graph(x, edge_index, num_nodes)
            
            # 更新缓存
            self.cached_weight = weight
            self.cached_features_weighted = features_weighted
            self.cached_eg = eg
            self.cached_en = x.clone()
        else:
            # 使用缓存
            eg = self.cached_eg
        
        # 计算异常分数
        score = self.discriminator(x.detach(), eg.detach())
        return score
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        编码函数，返回聚合后的特征
        
        Args:
            x: 节点特征
            edge_index: 边索引
            
        Returns:
            聚合特征
        """
        num_nodes = x.size(0)
        
        # 检查缓存
        if (self.cached_weight is None or 
            self.cached_weight.size(0) != num_nodes or
            self.cached_en is None or
            not torch.equal(self.cached_en, x)):
            
            # 重新计算
            weight, features_weighted, eg = self._preprocess_graph(x, edge_index, num_nodes)
            
            # 更新缓存
            self.cached_weight = weight
            self.cached_features_weighted = features_weighted
            self.cached_eg = eg
            self.cached_en = x.clone()
        
        return self.cached_eg
    
    def get_anomaly_scores(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        获取异常分数
        
        Args:
            x: 节点特征
            edge_index: 边索引
            
        Returns:
            异常分数
        """
        return self.forward(x, edge_index)