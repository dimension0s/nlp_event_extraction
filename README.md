# nlp_event_extraction
基于pytorch的事件抽取任务
对于多标签且不均衡的分类任务，通常采取的方法是定义focal loss，通过减少简单样本的损失并放大困难样本的损失，来起到均衡样本的效果。
我在该任务中使用了2种方案：
1.使用focal loss函数，alpha=0.5,gamma=2.2,epoch=15,batch_size=4,lr=1e-5；结果：accuracy:0.9245,f1:0.5851
2.直接调用nn.CrossEntropyLoss()损失，epoch=3,lr-1e-5,batch_size=4;结果：accuracy:0.9256,f1:0.6257
经过比对，建议选择第二种方案，效率更高一些
