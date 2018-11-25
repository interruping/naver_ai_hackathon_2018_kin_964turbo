# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import argparse
import os

import numpy as np

from keras.models import Sequential
from keras.layers import Input
from keras.layers import GRU 
from keras.models import Model
from keras.models import model_from_json
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from konlpy_set import KinQueryDatasetVer2, preprocess, get_embedding_dim
from util import ManDist
import pickle

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        #Keras 모델
        lstm_model = model['lstm_model']
        lstm_json = lstm_model.to_json()
        with open(os.path.join(dir_name, 'lstm_model'), "w") as json_file:
            json_file.write(lstm_json)
        lstm_model.save_weights(os.path.join(dir_name, "lstm_model.h5"))


    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):


        #LSTM 로드
        lstm_model = None
        with open(os.path.join(dir_name, 'lstm_model'), "r") as json_file:
            lstm_model = json_file.read()

        model['lstm_model'] = model_from_json(lstm_model, custom_objects={"ManDist":ManDist})
        model['lstm_model'].load_weights(os.path.join(dir_name, "lstm_model.h5"))

        model['lstm_model'].compile(loss='mean_squared_error', optimizer='adam')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        gram_y, first_x, second_x = preprocess(raw_data, config.strmaxlen)

        #LSTM Predict
        lstm_model = model['lstm_model']
        lstm_predic = lstm_model.predict([first_x, second_x])

        final_y = lstm_predic

        clipped = np.array(final_y > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(final_y.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default="0")

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=500)
    args.add_argument('--batch', type=int, default=2500)
    args.add_argument('--strmaxlen', type=int, default=20)
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--n_hidden', type=int, default=15)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/kin/'

    #analysis_data_feature()

    x = Sequential()
    x.add(GRU(config.n_hidden, input_shape=(config.strmaxlen, get_embedding_dim())))
    shared_model = x

    left_input = Input(shape=(config.strmaxlen, get_embedding_dim()), dtype='float32')
    right_input = Input(shape=(config.strmaxlen, get_embedding_dim()), dtype='float32')

    malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
    lstm_model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    bind_model({'lstm_model': lstm_model}, config=config)

    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        dataset = KinQueryDatasetVer2(DATASET_PATH, config.strmaxlen)
        print("loaded complete")
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(int(config.iteration) ,config.epochs):
            avg_loss = 0.0
            for i, (ngram_x, first_x, second_x, labels_y) in enumerate(_batch_loader(dataset, config.batch)):
                history = lstm_model.fit([first_x, second_x], labels_y, epochs=1, verbose=0)
                loss = 0
                if history:
                    loss = history.history['loss'][0]

                print('Batch : ', i + 1, '/', one_batch_size,
                         ', BCE in this minibatch: ', float(loss))
                avg_loss += float(loss)
            print('epoch:', epoch, ' train_loss:', float(avg_loss/one_batch_size))
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/one_batch_size),  step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # # 로컬 테스트 모드일때 사용합니다
    # # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        lstm_json = lstm_model.to_json()
        with open(os.path.join('./', 'lstm_model'), "w") as json_file:
            json_file.write(lstm_json)
        lstm_model.save_weights(os.path.join('./', "lstm_model.h5"))

        with open(os.path.join('./', 'lstm_model'), "r") as json_file:
            lstm_model = json_file.read()

        load_model = model_from_json(lstm_model, custom_objects={"ManDist": ManDist})
        load_model.load_weights(os.path.join('./', "lstm_model.h5"))

        load_model.compile(loss='mean_squared_error', optimizer='adam')

        with open(os.path.join(DATASET_PATH, 'train/test_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)