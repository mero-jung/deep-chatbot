import sys
sys.path.append('/content/drive/MyDrive/deep-chatbot/')

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing

intent_labels = {0: "인사", 1: "질문"}

# 의도 분류 모델 불러오기
model = load_model('drive/MyDrive/deep-chatbot/models/intent/intent_model.h5')

query = "데이터베이스"

from utils.Preprocess import Preprocess
p = Preprocess(word2index_dic='drive/MyDrive/deep-chatbot/train_tools/dict/chatbot_dict.bin',
               userdic='drive/MyDrive/deep-chatbot/utils/user_dic.tsv')
pos = p.pos(query)
keywords = p.get_keywords(pos, without_tag=True)
seq = p.get_wordidx_sequence(keywords)
sequences = [seq]

# 단어 시퀀스 벡터 크기
from config.GlobalParams import MAX_SEQ_LEN
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

predict = model.predict(padded_seqs)
predict_class = tf.math.argmax(predict, axis=1)
print(query)
print("의도 예측 점수 : ", predict)
print("의도 예측 클래스 : ", predict_class.numpy())
print("의도  : ", intent_labels[predict_class.numpy()[0]])

