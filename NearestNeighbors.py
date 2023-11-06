import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_index(data_vectors):
    # 创建NearestNeighbors模型
    model = NearestNeighbors(n_neighbors=50, algorithm='auto',n_jobs=-1)
    model.fit(data_vectors)  # 使用大规模向量库来拟合模型
    
    return model


def find_k_similar(query,index,k=50):
    distances, indices = index.kneighbors(query)
    return indices