

import numpy as np
import sklearn


test = "test_cluster"


if(test == "knn"):

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target

    X_train,X_test,y_train,y_test = train_test_split(iris_X,iris_y,test_size=0.3)
    knn = KNeighborsClassifier()
    knn.fit(X_train,y_train)

    print(knn.predict(X_test))

    print(y_test)

    print(knn.score(X_test,y_test))


if(test == "makedata"):

    from sklearn import datasets

    X,y = datasets.make_regression(n_samples=300,n_features=1,n_targets=1,noise=0)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X,y)
    plt.show()


if(test == "model_attr"):

    from sklearn import datasets
    from sklearn.linear_model import LinearRegression

    boston = datasets.load_boston()
    X = boston.data
    y = boston.target

    model = LinearRegression()
    model.fit(X,y)
##    model.predict(X[:4,:])

    print("coef is ",model.coef_)
    print('intercept is ',model.intercept_)
    print(model.get_params())
    print(model.score(X,y))


if(test == "preprocess"):

    from sklearn import preprocessing
    import numpy as np
    from sklearn.datasets.samples_generator import make_classification
    from sklearn.svm import SVC
    import matplotlib.pyplot as plt

    X,y=make_classification(n_samples=300,n_features=2,n_redundant=0,n_informative=2,random_state=22,n_clusters_per_class=1,scale=100)
    plt.figure
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.show()

    X=preprocessing.minmax_scale(X)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    clf=SVC()
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))


if(test == "K-fold-cross-validation"):

    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score#引入交叉验证
    import  matplotlib.pyplot as plt

    ## 加载数据
    iris=load_iris()
    X=iris.data
    y=iris.target

    ## k折交叉验证
    k_range=range(1,31)
    k_score=[]
    for k in k_range:
        knn=KNeighborsClassifier(n_neighbors=k)
        scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')#for classfication
        k_score.append(scores.mean())
    plt.figure()
    plt.plot(k_range,k_score)
    plt.xlabel('Value of k for KNN')
    plt.ylabel('CrossValidation accuracy')
    plt.show()


if(test == "overfitting"):

    pass


#############################################################################


if(test == "test_classification"): # 分类

    from sklearn import svm
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

    model = svm.SVC()
    model.fit(X_train,y_train)

    print(model.score(X_test,y_test))

    model.predict(X_test)
    

if(test == "test_regression"): # 回归

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_boston

    boston = load_boston()
    X = boston.data
    y = boston.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

    model = LinearRegression()
    model.fit(X_train,y_train)

    print(model.score(X_test,y_test))

    model.predict(X_test)


if(test == 'test_cluster'): # 聚类

    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

    model = KMeans(n_clusters = 3,random_state = 0)
    model.fit(X)

    model.predict(X[0:2])



if(test == "test_decomposition"): # 降维

    from sklearn.decomposition import PCA
    from sklearn.datasets import load_digits

    X,y = load_digits(return_X_y = True)

    model = PCA(n_components = 0.95)
    model.fit_transform(X)

    
