import numpy
import torch 
import csv 




def read(path = '', tensor  = False):
    with open(path, newline= "") as file:
        reader = csv.reader(file)
        data = []
        i = 0
        for row in reader:
            i += 1
            try:
                row = [float(x) for x in row]
                data.append(row)
            except:
                continue

    data = numpy.array(data)[:,1:]
    data = data/numpy.sum(data,axis = 1, keepdims=True)

    feature1 = data[:-2]
    feature2 = data[1:-1]
    feature = numpy.concatenate([feature1, feature2], axis=1)
    label = numpy.argmin(data[2:],axis = 1)


    if tensor:
        return torch.tensor(feature).float(), torch.tensor(label).float()
    return feature, label

if __name__ == '__main__':
    feature, label = read('./data/b-server.csv')
    print(feature)
    print(label)