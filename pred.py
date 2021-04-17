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
data.append( [32.03, 34.10, 33.87] )    #8
data.append( [28.28, 36.01, 35.71] )    #9
# data.append( [] )

data = numpy.array(data)
f1 = data[:-1]
f2 = data[1:]
feature = numpy.concatenate([f1,f2],axis = 1)

test_feature = torch.tensor(feature).float()/100



if __name__ == '__main__':
    times = 10000
    DNN = NN()
    DNN.load_state_dict(torch.load('./model.pickle'))
    pred = numpy.zeros((times, test_feature.shape[0]))
    for time in range(times):
        with torch.no_grad():           #测试
            x_test = DNN(test_feature)   #使用整个测试集一同测试
            pred[time] = x_test.argmax(1)
    prob = numpy.zeros((test_feature.shape[0], 3))
    for day in range(test_feature.shape[0]):
        prob[day,:] = \
            [ numpy.sum(pred[:,day]==0)/times, \
                numpy.sum(pred[:,day]==1)/times, \
                    numpy.sum(pred[:,day]==2)/times ]
    for p, day in zip(prob,range(len(pred.T))):
        print('The %d-th pred result: \t 30= %f%%;  \t 40= %f%%; \t 50=%f%%; '\
            %(day + 4, p[0]*100, p[1]*100, p[2]*100), end = '\t\t' )
        print('Expected return: \t 30= %f;  \t 40= %f;\t choice= %d'\
            %( p[0]*60+p[1]*15  , p[0]*20+p[1]*80, 30 if p[0]*60+p[1]*15>p[0]*20+p[1]*80 else 40 ))
        pass
    
    pass

