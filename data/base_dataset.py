import sys
import collections

def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf8', errors='ignore')


def read_file(file_name):
    data = []
    with open_file(file_name) as f:
        for line in f:
            try:
                label, content = line.split('\t')
                data.append([content.replace('\n', ''), label])
            except:
                pass

    return data


def tokenizer(text):
#     return [tok for tok in jieba.cut(text, cut_all=False)]
    return list(text)


# 根据词典，将数据转换成特征向量。
def encode_samples(tokenized_samples, vocab):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in vocab.token_to_idx:
                feature.append(vocab.token_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features


def pad_samples(features, maxlen=500, padding=0):
    padded_features = []
    for feature in features:
        if len(feature) > maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            # 添加 PAD 符号使每个序列等长（长度为 maxlen ）。
            while len(padded_feature) < maxlen:
                padded_feature.append(padding)
        padded_features.append(padded_feature)
    return padded_features


def count_token(train_tokenized):
    token_counter = collections.Counter()

    for sample in train_tokenized:
        for token in sample:
            if token not in token_counter:
                token_counter[token] = 1
            else:
                token_counter[token] += 1

    return token_counter