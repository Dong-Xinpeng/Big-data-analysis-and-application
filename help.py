import numpy as np
import pickle 

def check_query_emb():
    temp = np.load("./data/query_emb.npy")
    print("## query_emb.npy")
    print("type:",type(temp))
    print("shape:",temp.shape)
    # print(temp[0])

def check_labels_500():
    file = open("./data/labels_500.pkl","rb")
    print("## labels_500.pkl")
    temp  = pickle.load(file)
    print("type:",type(temp))
    print("shape:",len(temp),"*",len(temp[0]))
    # print(temp)

def check_gallery_emb():
    temp = np.load("./data/gallery_emb.npy")
    print("## gallery_emb.npy")
    print("type:",type(temp))
    print("shape:",temp.shape)
    # print(temp[0])


check_query_emb()
check_labels_500()
check_gallery_emb()