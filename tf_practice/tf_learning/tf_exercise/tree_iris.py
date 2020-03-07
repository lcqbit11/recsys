import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split

iris=load_iris()
cur_state = np.random.get_state()
np.random.shuffle(iris.data)
np.random.set_state(cur_state)
np.random.shuffle(iris.target)

train_X, test_X, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)


# test_idx=[0,50,100] #三朵预留出来做测试的花
#
# train_target=np.delete(iris.target,test_idx,0) #训练模型不包含三朵花
# train_data=np.delete(iris.data,test_idx,0) #训练模型不包含三朵花
#
# test_target=iris.target[test_idx]
# test_data=iris.data[test_idx]

#用数据训练计算机
clf=tree.DecisionTreeClassifier()  #这里使用了决策树分类器
clf.fit(train_X,train_y)

print(test_y) #植物学家对三朵花分类的看法
print(clf.predict(test_X)) #计算机对三朵花分类的看法
score = clf.score(test_X, test_y)
print(score)