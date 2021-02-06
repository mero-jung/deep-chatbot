import pickle

f = open("drive/MyDrive/deep-chatbot/train_tools/dict/chatbot_dict.bin", "rb")
word_index = pickle.load(f)
f.close()

print(word_index['OOV'])
print(word_index['교수'])
print(word_index['다이어그램'])
print(word_index['트랜잭션'])
print(word_index['정규화'])
