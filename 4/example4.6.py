from sklearn.datasets import fetch_20newsgroups  # 导入fetch_20newsgroups函数，用于获取新闻数据集
from sklearn.naive_bayes import MultinomialNB  # 导入MultinomialNB朴素贝叶斯分类器
from sklearn.feature_extraction.text import TfidfTransformer  # 导入TfidfTransformer，用于计算TF-IDF特征
from sklearn.feature_extraction.text import CountVectorizer  # 导入CountVectorizer，用于将文本转换为词频向量

category_map = {'talk.politics.misc': 'Politics',  # 定义新闻类别的映射关系
                'rec.autos': 'Autos',
                'rec.sport.hockey': 'Hockey',
                'sci.electronics': 'Electronics',
                'sci.med': 'Medicine'}

training_data = fetch_20newsgroups(subset='train',  # 获取训练数据集
                                   categories=category_map.keys(),
                                   shuffle=True,
                                   random_state=5)

count_vectorizer = CountVectorizer()  # 创建CountVectorizer对象，用于将文本转换为词频向量
train_tc = count_vectorizer.fit_transform(training_data.data)  # 对训练数据进行词频向量化

tfidf = TfidfTransformer()  # 创建TfidfTransformer对象，用于计算TF-IDF特征
train_tfidf = tfidf.fit_transform(train_tc)  # 对词频向量进行TF-IDF转换

input_data = [  # 待分类的输入数据
    'Be sure to take medicine when you are ill',
    'The player made a mistake in passing the ball',
    'Be sure to fasten your seat belt when you drive a car',
    'Sensing technology has been well applied in this device'
]

classifier = MultinomialNB().fit(train_tfidf, training_data.target)  # 创建并训练MultinomialNB分类器

input_tc = count_vectorizer.transform(input_data)  # 对输入数据进行词频向量化

input_tfidf = tfidf.transform(input_tc)  # 对词频向量进行TF-IDF转换

predictions = classifier.predict(input_tfidf)  # 对输入数据进行分类预测

for sent, category in zip(input_data, predictions):  # 遍历输入数据和预测结果
    print('\nInput sentence:', sent, '\nPredicted category:', \
          category_map[training_data.target_names[category]])  # 打印输入句子和预测的类别名称
