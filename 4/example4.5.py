import chunker as chunker  # 导入自定义的chunker模块
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer  # 导入CountVectorizer用于构建文档-词频矩阵
from nltk.corpus import brown
from text_chunker import chunker  # 导入自定义的chunker函数

input_data = ' '.join(brown.words()[:5600])  # 获取brown语料库中的前5600个单词，并将它们拼接成一个字符串
chunk_size = 900  # 指定块的大小为900
text_chunks = chunker(input_data, chunk_size)  # 调用自定义的chunker函数将输入数据分块

chunks = []  # 存储分块数据的列表
for count, chunk in enumerate(text_chunks):  # 遍历每个分块的索引和内容
    d = {'index': count, 'text': chunk}  # 创建一个字典，包含索引和分块文本内容
    chunks.append(d)  # 将字典添加到chunks列表中

count_vectorizer = CountVectorizer(min_df=7, max_df=18)  # 创建一个CountVectorizer对象，用于构建文档-词频矩阵
document_term_matrix = count_vectorizer.fit_transform([chunk['text'] for chunk in chunks])  # 将分块文本列表转换为文档-词频矩阵

vocabulary = np.array(count_vectorizer.get_feature_names())  # 获取词汇表中的词语列表

chunk_names = []  # 存储分块名称的列表
for i in range(len(text_chunks)):  # 遍历每个分块的索引
    chunk_names.append('Chunk ' + str(i + 1))  # 构建分块的名称，形如"Chunk 1"

print("\nDocument Term Matrix:")
formatted_text = '{:>9}' * (len(chunk_names) + 1)  # 构建格式化的表头字符串
print('\n', formatted_text.format('Word', *chunk_names),'\n', '*'*76)  # 打印表头

for word, item in zip(vocabulary, document_term_matrix.T):  # 遍历词汇表和矩阵的列（按列迭代）
    output = [word] + [str(freq) for freq in item.data]  # 构建每一行的输出内容
    print(formatted_text.format(*output))  # 打印格式化的行数据
