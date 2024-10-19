# 2.模型构建:对于多标签任务，尝试使用多头线性层
# 好处：
# 对于分布不均的标签，使用多头可以分别处理不同特征空间的标签

from transformers import AutoConfig
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from label_dict import label2id,id2label
from collate_fn import checkpoint,tokenizer,batch_X,batch_y
from device import device


class BertForEventExtraction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 使用多头分类器（多层线性层或高维变换）
        self.hidden_size = 768
        self.num_labels = len(id2label)

        self.cls = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(1024, self.num_labels))
        self.post_init()  # 后处理

    def forward(self, x):
        bert_output = self.bert(**x)
        sequence_output = bert_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.cls(sequence_output)
        return logits


config = AutoConfig.from_pretrained(checkpoint)
model = BertForEventExtraction.from_pretrained(checkpoint, config=config).to(device)
# 会输出提示信息：告诉你新添加的层需要进行训练

print(model)
# 测试
outputs = model(batch_X.to(device))
print(outputs.shape)