import pickle

f = open("drive/MyDrive/deep-chatbot/train_tools/dict/chatbot_dict.bin", "rb")
word_index = pickle.load(f)
f.close()

print(word_index['OOV'])
print(word_index['데이터'])
print(word_index['설명'])
#print(word_index['DB'])
#print(word_index['Database'])