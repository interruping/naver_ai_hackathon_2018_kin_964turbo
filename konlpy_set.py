from konlpy.tag import Twitter
import numpy as np
import os
from gensim.models import word2vec
from gensim.models.wrappers import FastText

class KinQueryDatasetVer2:
    """
        지식인 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    '''
    문장의 길이 sentence_length
    문장 길이의 차이 | a - b | abs_reduce_length
    일치하는 명사 수 match_noun_count
    많이 사용된 명사
    :return:
    '''
    """
    def __init__(self, dataset_path: str, max_length: int):
        """

        :param dataset_path: 데이터셋 root path
        :param max_length: 문자열의 최대 길이
        """
        # 데이터, 레이블 각각의 경로
        queries_path = os.path.join(dataset_path, 'train', 'train_data')
        labels_path = os.path.join(dataset_path, 'train', 'train_label')

        # 지식인 데이터를 읽고 preprocess까지 진행합니다
        with open(queries_path, 'rt', encoding='utf8') as f:
            self.ngram_x, self.first_x, self.second_x = preprocess(f.readlines(), max_length)
        # 지식인 레이블을 읽고 preprocess까지 진행합니다.
        with open(labels_path) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])

    def __len__(self):
        """

        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.first_x)

    def __getitem__(self, idx):
        """

        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.ngram_x[idx, :], self.first_x[idx, :], self.second_x[idx, :], self.labels[idx]

def ngram(s, num):
    res = []
    slen = len(s) - num + 1
    for i in range(slen):
        ss = s[i:i+num]
        res.append(ss)
    return res

def diff_ngram(sa, sb, num):
    a = ngram(sa, num)
    b = ngram(sb, num)
    r = []
    cnt = 0
    for i in a:
        for j in b:
            if i == j:
                cnt += 1
                r.append(i)
    alen = len(a)
    if alen == 0:
        alen = 1

    return cnt / alen, r

def get_embedding_dim():
    return 300


def preprocess(data: list, max_length: int):
    """
    """
    ngram_result = np.zeros((len(data), 1), dtype=np.float32)
    first_sentences_word2vec = np.zeros(shape=(len(data), max_length, get_embedding_dim()))
    second_sentences_word2vec = np.zeros(shape=(len(data), max_length, get_embedding_dim()))
    twitter = Twitter()

    vocabulary = dict()
    inverse_vocabulary = ['<unk>']
    voca_model = FastText.load_fasttext_format('/pretrained/wiki.ko.bin')
    for root_i, sentense in enumerate(data):

        first_s, second_s = sentense.split('\t')

        #ngram 유사도 측정
        r, _ = diff_ngram(first_s, second_s, 2)
        ngram_result[root_i] = r


        # ( '단어' : '품사' ) 배열
        first_pos = twitter.pos(first_s)
        second_pos = twitter.pos(second_s)

        filtered_first = []
        filtered_second = []

        # 형태소 분석 및 처리
        # 첫번째 문장 처리
        for pos_word in first_pos:
            if not pos_word[1] in ['Josa', 'Eomi', 'Punctuation']:
                filtered_first.append(pos_word[0])
        # 두번째 문장 처리
        for pos_word in second_pos:
            if not pos_word[1] in ['Josa', 'Eomi', 'Punctuation']:
                filtered_second.append(pos_word[0])

        # 첫 번째 문장 원 핫 처리
        first_q2n = []
        input_first_index = 0
        for i, word in enumerate(filtered_first):
            if not input_first_index < max_length:
                break;

            if word in voca_model:
                first_sentences_word2vec[root_i][input_first_index][:] = voca_model[word]
                input_first_index+=1
        # 두 번째 문장 원 핫 처리
        second_q2n = []
        input_second_index = 0
        for i, word in enumerate(filtered_second) :
            if not input_second_index < max_length:
                break;

            if word in voca_model:
                second_sentences_word2vec[root_i][input_second_index][:] = voca_model[word]
                input_second_index+=1

        if root_i % 1000 == 0:
            print('iter:' + str(root_i) + "/" + str(len(data)))

    del voca_model

    return ngram_result , first_sentences_word2vec, second_sentences_word2vec

