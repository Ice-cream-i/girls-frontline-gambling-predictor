from torch.autograd.grad_mode import F
from data.predata import read
from DNN import NN
import torch 
import numpy 
data = []

#录入数据，请勿删除已有数据，直接在后面按格式加就行。
data.append( [31.09, 31.39, 37.53] )    #2
data.append( [25.77, 39.48, 34.74] )    #3
data.append( [29.48, 29.89, 40.64] )    #4
data.append( [29.65, 34.60, 35.75] )    #5
data.append( [35.34, 30.83, 33.83] )    #6
data.append( [31.20, 31.12, 37.67] )    #7
# data.append( [] )

data = numpy.array(data)
f1 = data[:-1]
f2 = data[1:]
feature = numpy.concatenate([f1,f2],axis = 1)

test_feature = torch.tensor(feature).float()/100



if __name__ == '__main__':
    DNN = NN()
    DNN.load_state_dict(torch.load('./model.pickle'))
    with torch.no_grad():           #测试
        x_test = DNN(test_feature)   #使用整个测试集一同测试
        pred = x_test.argmax(1)
        for p, day in zip(pred,range(len(pred))):
            print('The %d-th pred result = %d'%(day + 4, p*10 + 30))

