from os import name

from numpy.core.defchararray import mod
from data.predata import read
from sklearn import naive_bayes
from sklearn import svm
import numpy 

def acc(path = ''):
    feature, label = read(path)

    pred = model.predict(feature)
    acc = numpy.sum(pred == label)/label.shape[0]

    return acc



if __name__ == '__main__':
    # model = naive_bayes.GaussianNB()
    # model = svm.SVC()


    feature, label = read('./data/b-server.csv')

    model.fit(feature,label)
    pred = model.predict(feature)

    ACC = acc('./data/b-server.csv')
    print(ACC)

    ACC = acc('./data/An-server.csv')
    print(ACC)

