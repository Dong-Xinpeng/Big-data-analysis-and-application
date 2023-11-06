import numpy as np
import pickle 
from sklearn.neighbors import NearestNeighbors
import time

data = np.load('./gallery_emb.npy')
query = np.load('./query_emb.npy')

# 定义k值，即要查询的最相似的K个向量
K = 50

# 创建NearestNeighbors模型
model = NearestNeighbors(n_neighbors=K, algorithm='auto')
model.fit(data)  # 使用大规模向量库G来拟合模型


# 进行查询
# distances是N个查询向量的最相似的K个向量的距离
# indices是N个查询向量的最相似的K个向量的索引
start_time = time.time()
distances, indices = model.kneighbors(query)


print(f"花费的时间: {(time.time() - start_time)/500}")

print("{")
# 输出结果
for i in range(500):
    print(f"{i}:[{','.join(np.char.mod('%d',indices[i]))}],")
    
print("}")

# with open("./labels_500.pkl",'rb') as file:
#     data = pickle.load(file)
#     print(data)

