# 3.2)验证函数

import numpy
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from device import device
import torch
from label_dict import label2id,id2label



def test_loop(dataloader, model, mode='Valid'):
    assert mode in ['Valid', 'test']
    true_labels, true_predictions = [], []

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)  # [batch_size, sequence_length, num_labels]
            predictions = pred.argmax(dim=-1).cpu().numpy()  # [batch_size, sequence_length]
            labels = y.cpu().numpy()

            # 将真实标签平铺
            for label_seq, pred_seq in zip(labels, predictions):
                # 确保 `label_seq` 和 `pred_seq` 不是列表，而是单个标签
                for true_label, pred_label in zip(label_seq, pred_seq):
                    if true_label != -100:  # 排除 mask 掩码标签
                        true_labels.append(id2label[true_label])
                        true_predictions.append(id2label[pred_label])
                # true_pred = [id2label[p] for p,l in zip(prediction,label) if l!=-100]
                # true_label = [id2label[l] for l in label if l!=-100]
                # true_predictions.append(true_pred)
                # true_labels.append(true_label)

    # 转换为 id 以便计算
    true_ids = [label2id[l] for l in true_labels]
    pred_ids = [label2id[p] for p in true_predictions]

    # 计算性能指标
    accuracy = accuracy_score(true_ids, pred_ids)
    precision, recall, f1, _ = precision_recall_fscore_support(true_ids, pred_ids, average='macro')

    print(f"accuracy:{accuracy:.4f},precision:{precision:.4f},recall:{recall:.4f},f1-score: {f1:.4f}")

    return accuracy, precision, recall, f1