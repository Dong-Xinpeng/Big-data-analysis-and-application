# 慢到离谱,有待改进
import time
import numpy as np
import pickle 
from datasketch import MinHashLSHForest, MinHash

def build_index(data_vectors):
    # print(type(data_vectors[0]))
    # print(data_vectors[0])
    # print(len(data_vectors))
    all_len  = len(data_vectors)
    forest = MinHashLSHForest(num_perm=128)
    for vec_index in range(len(data_vectors)):
        # indexing pocess
        if(vec_index%10000==0):
            print("index_building:{}/{}".format(vec_index,all_len))
        m = MinHash(num_perm=128)
        for d in data_vectors[vec_index]:
            m.update(str(d).encode('utf8'))
        forest.add(vec_index, m)     
    forest.index()
    with open('bad_lsh_index.pkl', 'wb') as file:
        pickle.dump(forest, file)
    return forest


def find_k_similar(query, index, k=50):
    result = []
    for each_query in query:
        m = MinHash(num_perm=128)
        for d in each_query:
            m.update(str(d).encode('utf8'))
        result.append(index.query(m, k))
    return result


if __name__ == '__main__':
    # 是否重新构建forest
    re_train = True

    query = np.load("./data/query_emb.npy")
    print("query loading finished")

    if re_train:
        raw_data = np.load("./data/gallery_emb.npy")
        print("data loading finished")
        forest = build_index(raw_data)
        print("building index finished,ready to query")
    else:
        file = open("./data/labels_500.pkl","rb")
        forest  = pickle.load(file)
        print("use old forest, ready to query")

    start_time = time.perf_counter()
    res = find_k_similar(query,forest)
    end_time = time.perf_counter()

    execution_time = end_time - start_time

    print(res)  # 目前返回的是key，如何显示原文
    print("query time cost:{}".format(execution_time))
