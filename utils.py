import numpy as np

def make_dummy(data,key):
    temp = np.zeros([data.shape[0],len(key)])
    idx = list(map(lambda x:list(key).index(x),data))
    temp[np.arange(data.shape[0]),idx]=1
    return temp

def test_dummy(data,key):
    temp = np.zeros(len(key))
    idx = list(key).index(data)
    temp[idx]=1
    return temp
