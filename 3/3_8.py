# 导入matplotlib库中的pyplot模块并简写为plt，用于绘图
import matplotlib.pyplot as plt
# 从sklearn库中导入datasets模块，用于加载数据集
from sklearn import datasets
# 从sklearn库中导入LinearRegression类，用于线性回归算法
from sklearn.linear_model import LinearRegression
# 导入pandas库并简写为pd，用于数据处理
import pandas as pd
# 从sklearn库中导入train_test_split函数，用于划分训练集和测试集
from sklearn.model_selection import train_test_split
# 导入DataFrame类，用于创建和处理数据框
from pandas import DataFrame
# 从sklearn库中导入r2_score函数，用于计算R^2评估指标

# 使用datasets模块中的load_boston函数加载波士顿房价数据集
bos = datasets.load_boston()
# 获取数据集的特征数据，存储在变量x中
x = bos.data
# 获取数据集的目标值（房价），存储在变量y中
y = bos.target
# 使用DataFrame类将特征数据x转换为数据框，并指定列名为数据集的特征名称
df = pd.DataFrame(x, columns=bos.feature_names)
# 定义要删除的特征列表
features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'AGE', 'DIS', 'RAD', 'TAX', 'B']
# 使用drop函数删除数据框df中features列表中对应的列，并返回删除后的结果
tmp = df.drop(features, axis=1)

# 初始化一个空列表，用于存储需要删除的行号
tmp_row = []
# 遍历目标值y的每个元素
for i in range(len(y)):
    # 如果目标值等于50，将当前行号i添加到tmp_row列表中
    if y[i] == 50:
        tmp_row.append(i)

# 使用drop函数删除数据框tmp中tmp_row列表中对应的行，并返回删除后的特征数据x
x = tmp.drop(tmp_row)
# 使用DataFrame类将目标值y转换为数据框，并使用drop函数删除tmp_row列表中对应的行，并返回删除后的目标值y
y = pd.DataFrame(y).drop(tmp_row)

# 使用train_test_split函数划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.20)

# 创建LinearRegression实例lr
lr = LinearRegression()
# 使用训练数据进行线性回归模型的参数估计
lr.fit(X_train, y_train)

# 打印截距项
print(lr.intercept_)
# 打印线性回归模型的系数
print(lr.coef_)

# 使用训练好的线性回归模型进行测试集的房价预测
y_pred = lr.predict(X_test)

# 创建一个图形对象fig，并设置其大小为12x6
fig = plt.figure(figsize=(12, 6))
# 绘制蓝色的实际房价曲线
plt.plot(range(y_test.shape[0]), y_test, color='blue', linewidth=1.5, linestyle='-')
# 绘制红色的预测房价曲线
plt.plot(range(y_test.shape[0]), y_pred, color='red', linewidth=1.5, linestyle='-.')
# 设置图例
plt.legend(["source", "predict"])
# 显示图形
plt.show()

# 计算R^2评估指标并打印结果
score = r2_score(y_test, y_pred)
print(score)
