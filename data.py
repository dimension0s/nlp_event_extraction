# 1.数据集处理
# 1.1）加载数据集

import json
from torch.utils.data import Dataset

categories = set()

class EventData(Dataset):
    def __init__(self,data_file):
        self.data = self.load_data(data_file)

    def load_data(self,data_file):
        Data = {}
        with open(data_file,'r',encoding='utf-8') as f:
            for idx, line in enumerate(f):
                item = json.loads(line.strip())  # 注意解析为json对象，不可省略
                text = item['text']  # 提取文本
                event_list = item['event_list']  # 提取事件列表
                for event in event_list:
                    labels = []
                    argument_end_idx = 0
                    event_type = event['event_type']  # 提取事件类型
                    arguments = event['arguments']  # 提取事件论元
                    for argument in arguments:
                        event_combo = event_type+'-'+argument['role']  # 实体组合，比如：组织关系-裁员-裁员人数
                        argument_end_idx = argument['argument_start_index']+len(argument['argument'])-1
                        # 构造标签：（事件类型-角色, 论元, 论元起始位置，论元结束位置）
                        labels.append([event_combo,
                                       argument['argument'],
                                       argument['argument_start_index'],
                                       argument_end_idx])
                        categories.add(event_combo)  # 添加实体类别，共217种
                Data[idx] = {
                    'text': text,
                    'event_type': event_type,
                    'labels': labels,}

        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = EventData("data/train.json")
valid_data = EventData("data/dev.json")

print(len(train_data))
print(len(valid_data))

# 打印数据集用于检查
print("train_dataset[0]:\n", train_data[0])
print("valid_dataset[0]:\n", valid_data[0])
