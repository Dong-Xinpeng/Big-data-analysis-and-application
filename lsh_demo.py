# todo：也许该package/method不合适（将int编为str再encode，离谱）

from datasketch import MinHashLSHForest, MinHash

# data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
#         'estimating', 'the', 'similarity', 'between', 'datasets']
# data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
#         'estimating', 'the', 'similarity', 'between', 'documents']
# data3 = ['minhash', 'is', 'probability', 'data', 'structure', 'for',
#         'estimating', 'the', 'similarity', 'between', 'documents']

data1 = [1,2,3,4,5,6]
data2 = [10,2,3,4,5,6]
data3 = [1,2,3,4.5,5,6]

# Create MinHash objects
m1 = MinHash(num_perm=128)
m2 = MinHash(num_perm=128)
m3 = MinHash(num_perm=128)
for d in data1:
    m1.update(str(d).encode('utf8'))
for d in data2:
    m2.update(str(d).encode('utf8'))
for d in data3:
    m3.update(str(d).encode('utf8'))

# Create a MinHash LSH Forest with the same num_perm parameter
forest = MinHashLSHForest(num_perm=128)

# Add m2 and m3 into the index
forest.add("m2", m2)
forest.add("m3", m3)

# IMPORTANT: must call index() otherwise the keys won't be searchable
forest.index()

# Check for membership using the key
print("m2" in forest)
print("m3" in forest)

# Using m1 as the query, retrieve top 2 keys that have the higest Jaccard
result = forest.query(m1, 1)
print("Top 2 candidates", result)