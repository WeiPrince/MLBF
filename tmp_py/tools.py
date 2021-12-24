import random
import numpy as np
def getdistribution(data,sample_num):

    l=[]
    for i in range(len(data[0])):
        print(data[:,i])

        s = random.sample(list(data[:,i]), sample_num)
        l.append(len(set(s))/sample_num)
    return l
data=[[1,2,3,4,5],[2,2,3,4,5]]
print(getdistribution(np.array(data),2))