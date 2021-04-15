from torch.autograd.grad_mode import F
from data.predata import read
import torch 
import numpy 

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.acti    = torch.nn.ReLU()      #设置激活函数
        self.softmax = torch.nn.Softmax(1)

        self.linear =  torch.nn.Sequential(         #两层线性层
            torch.nn.Linear(6, 32),
            torch.nn.Dropout(0.1),
            torch.nn.BatchNorm1d(32),
            self.acti,
            torch.nn.Linear(32, 8),
            torch.nn.Dropout(0.1),
            torch.nn.BatchNorm1d(8),
            self.acti,
            torch.nn.Linear(8, 3),
            self.softmax
        )

    def forward(self, x):
        x = self.linear(x)              #线性层
        return x

if __name__ == '__main__':
    
    lr = 0.01           #设置学习率/mini-batch规模/epoch数目
    epochs = 50

    DNN = NN()   #初始化网络,使用GPU

    acc_test = []

    optimzer = torch.optim.Adam(DNN.parameters(), lr)   #Adam优化器
    loss_func = torch.nn.CrossEntropyLoss()             #交叉熵损失

    train_feature, train_labels = read('./data/b-server.csv', True)
    test_feature, test_labels = read('./data/An-server.csv', True)

    feature = torch.cat([train_feature,test_feature],dim=0)
    label = torch.cat([train_labels,test_labels],dim=0)

    test_num = 10

    train_feature = feature[:-test_num]
    train_labels  = label[:-test_num]
    test_feature = feature[-test_num:]
    test_labels = label[-test_num:]

    acc_best = 0
    for epoch in range(epochs):
        batch = train_feature
        x = DNN(batch)                          #计算网络输出
        pred = x.argmax(1)
        acc  = torch.mean((pred == train_labels).float())    #计算正确率
        target = train_labels.long() #取label
        loss = loss_func(x, target)             #计算损失
        optimzer.zero_grad()                    #梯度清零
        loss.backward()
        optimzer.step()
        #print('loss = %f'%loss, end = '\n')
        pred = x.argmax(1)
        acc  = torch.mean((pred == target).float()) #计算训练正确率
            
        with torch.no_grad():           #测试
            x_test = DNN(test_feature)   #使用整个测试集一同测试
            pred = x_test.argmax(1)
            acc  = torch.mean((pred == test_labels).float())    #计算正确率
            acc_test.append(acc)
            print('epoch %d finished! test_acc = %f'%(epoch, acc))
            target = test_labels.long()
            loss = loss_func(x_test, target)

            if acc > acc_best:
                # torch.save(DNN.state_dict(),'./model.pickle') 
                acc_best = acc
        

    best = numpy.argmax(acc_test)
    print('best epoch = %d'%best)
    print('best  acc  = %f'%acc_best)

    
    
    pass