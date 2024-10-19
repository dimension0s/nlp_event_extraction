# 1.2)构造标签映射字典

from data import categories

id2label = {0: 'O'}
for c in list(sorted(categories)):
    id2label[len(id2label)] = c

label2id = {v: k for k, v in id2label.items()}

print(id2label)
print(label2id)
print(len(label2id))