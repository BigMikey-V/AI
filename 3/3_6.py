# 从sklearn库中导入datasets模块，用于加载数据集
from sklearn import datasets
# 从sklearn库中导入SelectKBest类，用于特征选择
from sklearn.feature_selection import SelectKBest
# 从sklearn库中导入f_regression函数，用于特征选择评估指标
from sklearn.feature_selection import f_regression

# 使用datasets模块中的load_boston函数加载波士顿房价数据集
dataset = datasets.load_boston()
# 获取数据集的特征数据，存储在变量x中
x = dataset.data
# 获取数据集的目标值（房价），存储在变量y中
y = dataset.target
# 获取特征的名称，存储在变量names中
names = dataset.feature_names

# 创建SelectKBest实例s，使用f_regression作为特征选择的评估指标，选择k=3个最佳特征
s = SelectKBest(f_regression, k=3)

# 使用fit_transform函数对特征数据x和目标值y进行特征选择，并返回选择后的数据
s.fit_transform(x, y)

# 获取特征选择结果的布尔数组，表示哪些特征被选择
arr = s.get_support()

# 初始化变量i为0
i = 0
# 遍历特征选择结果的布尔数组
for t in arr:
    # 如果特征被选择（布尔值为True），则打印特征的名称
    if t:
        print(names[i])
    # 更新i的值，指向下一个特征
    i = i + 1
