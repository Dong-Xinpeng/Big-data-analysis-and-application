# 慢到离谱,有待改进

import numpy as np
import pickle 
from datasketch import MinHashLSHForest, MinHash

def build_index(data_vectors):
    # print(type(data_vectors[0]))
    # print(data_vectors[0])
    # print(len(data_vectors))
    forest = MinHashLSHForest(num_perm=128)
    for vec_index in range(len(data_vectors)):
        if(vec_index%100==0):
            print(vec_index)
        m = MinHash(num_perm=128)
        for d in data_vectors[vec_index]:
            m.update(str(d).encode('utf8'))
        forest.add(vec_index, m)     
    forest.index()
    return forest


def find_k_similar(query, index, k=50):
    one_query = query[0]
    m = MinHash(num_perm=128)
    for d in one_query:
        m.update(str(d).encode('utf8'))
    result = index.query(m, 50)
    return result


if __name__ == '__main__':
    raw_data = np.load("./data/gallery_emb.npy")
    query = np.load("./data/query_emb.npy")
    print("data loading finished")

    forest = build_index(raw_data)
    print("building index finished,ready to query")

    res = find_k_similar(query,forest)
    print(res)
