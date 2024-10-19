# 加载测试数据集
import json

def test_data_load(data_file):
    data = {}
    with open(data_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            text = item['text']
            text_id = item['id']

            data[idx] = {
                'text': text,
                'text_id': text_id,
            }
    return data


test_data = test_data_load("data/test.json")
for idx in range(5):
    print(f'sample {idx}:{test_data[idx]}')
