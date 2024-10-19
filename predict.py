# 5.预测函数与封装
from model import model
import torch
from device import device
from collate_fn import tokenizer
from label_dict import label2id,id2label
from test_data import test_data_load

model.load_state_dict(
    torch.load('epoch_15_valid_accuracy_0.9245_valid_f1_0.5851_weights.bin', map_location=torch.device('cpu'))
)

model.eval()


def predict(text, tokenizer, model):
    model.eval()

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors='pt', ).to(device)

    # 获取模型预测结果
    with torch.no_grad():
        outputs = model(inputs)  # [batch_size, seq_len, num_labels]

    # 获取每个token对应的预测标签
    predictions = outputs.argmax(dim=-1).cpu().numpy()[0]  # [seq_len]，取最大值所在的类别

    # 将token映射回原始文本
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].cpu().numpy()[0])  # 获取tokens
    predictions = predictions[:len(tokens)]  # 只保留与tokens对应的预测
    result = []

    # 将每个token的预测结果与文本对齐
    for token, label_id in zip(tokens, predictions):

        if token.startswith("##"):  # 跳过词片段
            continue
        label = id2label[label_id]
        # result.append((token, label))
        result.append(label)

    return result

test_data = test_data_load("E:\\NLP任务\\事件抽取\\data\\test1.json")
for i in range(3):
    data = test_data[i]
    preds = predict(data['text'],tokenizer,model)
    print(f"{data['text']}\npred:{preds}")
