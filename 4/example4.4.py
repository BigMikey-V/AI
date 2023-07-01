import numpy as np
from nltk.corpus import brown

# 定义一个分块函数，将输入的文本数据划分为指定大小的块
def chunker(input_data, N):
    input_words = input_data.split(' ')  # 将输入的文本按照空格分割为单词列表
    output = []  # 存储分块结果的列表
    cur_chunk = []  # 当前块的单词列表
    count = 0  # 记录当前块已经包含的单词数量
    for word in input_words:  # 遍历输入的单词列表
        cur_chunk.append(word)  # 将当前单词添加到当前块中
        count += 1  # 块中单词数量加1
        if count == N:  # 当块中的单词数量达到指定大小N时
            output.append(' '.join(cur_chunk))  # 将当前块转换为字符串，添加到输出列表中
            count, cur_chunk = 0, []  # 重置计数和当前块列表

    output.append(' '.join(cur_chunk))  # 将最后不足N个单词的块添加到输出列表中
    return output

if __name__ == '__main__':
    input_data = ' '.join(brown.words()[:6300])  # 获取brown语料库中的前6300个单词，并将它们拼接成一个字符串
    chunk_size = 800  # 指定块的大小为800
    chunks = chunker(input_data, chunk_size)  # 调用分块函数将输入数据分块
    print('\nNumber of text chunks =', len(chunks), '\n')  # 打印分块的数量
    for i, chunk in enumerate(chunks):  # 遍历每个分块
        print('Chunk', i + 1, '==>', chunk[:50])  # 打印每个分块的前50个字符
