import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from datasets import clinc150
# 初始化BERT模型和分词器
num_classes = 150
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 定义超参数
batch_size = 32
learning_rate = 1e-5
num_epochs = 10
# 持续学习过程
from datasets.clinc150 import split_data
tasks = split_data
for task, task_data in tasks.items():
    print("Task:", task)

    # 获取当前任务的训练数据和测试数据
    train_data = task_data['train']
    test_data = task_data['test']

    # 将数据转换为BERT输入格式
    train_inputs = tokenizer([data[0] for data in train_data], padding=True, truncation=True, return_tensors='pt')
    train_labels = torch.tensor([data[1] for data in train_data])
    test_inputs = tokenizer([data[0] for data in test_data], padding=True, truncation=True, return_tensors='pt')
    test_labels = torch.tensor([data[1] for data in test_data])

    # 创建数据加载器
    train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 设置优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 微调BERT模型
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # 在测试集上评估模型性能
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted_labels = torch.max(outputs.logits, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print("模型在测试集上的准确率:", accuracy)
