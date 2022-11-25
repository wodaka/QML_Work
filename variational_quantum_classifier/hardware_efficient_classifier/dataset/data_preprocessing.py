import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_mnist():
    #define the directory where mnist.npz is(Please watch the '\'!)
    path = r'.//dataset//mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'],f['y_train']
    x_test, y_test = f['x_test'],f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def filter_36(x, y): #y=3为true,y=6为false
    keep = (y == 3) | (y == 6)
    x, y = x[keep], y[keep]
    y = y == 3
    return x,y

def get_mnist():

    (x_train, y_train), (x_test, y_test) = load_mnist()

    x_train, y_train = filter_36(x_train, y_train)
    x_test, y_test = filter_36(x_test, y_test)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1] * x_train.shape[2]))
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))

    # #数据标准化
    sc = StandardScaler().fit(x_train) #计算均值和方差
    sc1 = StandardScaler().fit(x_test)

    X_std_train = sc.transform(x_train)
    X_std_test = sc1.transform(x_test)

    pca_x_train = PCA(6).fit_transform(X_std_train)
    pca_x_test = PCA(6).fit_transform(X_std_test)

    x_train = np.arctan(pca_x_train)
    x_test = np.arctan(pca_x_test)

    #标签从True,False转为1,0（3,1）（6,0）
    y_train = y_train + 0
    y_test = y_test + 0

    # print(pca_x_test[0])
    # print(x_test[0])

    return x_train,y_train,x_test,y_test