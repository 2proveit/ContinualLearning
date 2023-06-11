import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from datasets import clinc150
from datasets.clinc150 import clinc150_classes
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 初始化BERT模型和分词器
num_classes = 150
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# 定义标签映射
label_mapping = {label: idx for idx, label in enumerate(clinc150_classes)}
# 定义超参数
batch_size = 32
learning_rate = 1e-5
num_epochs = 10
# 保存实验结果
SVAE_PATH = './outputs/bert_vanilla'
logs = {}
if not os.path.exsits(SVAE_PATH):
    os.mkdir(SVAE_PATH)

# 持续学习过程
from datasets.clinc150 import split_data
tasks = split_data

# 在当前任务上训练
for task, task_data in tasks.items():
    print("Training on task:\t", task)

    # 获取当前任务的训练数据和测试数据
    train_data = task_data['train']

    # 将数据转换为BERT输入格式
    train_inputs = tokenizer([data[0] for data in train_data], padding=True, truncation=True, return_tensors='pt')
    train_labels = torch.tensor([label_mapping[data[1]] for data in train_data])
   
    # 创建数据加载器
    train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    

    # 设置优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 微调BERT模型
    model.train()
    pervious_epoch_loss = 1e9
    for epoch in range(num_epochs):
        print(f'epoch: {epoch}/{num_epochs - 1}')
        epoch_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            with torch.no_grad():
                loss += epoch_loss
            loss.backward()
            optimizer.step()
        print(f"task: {task}\t epoch: {epoch}/{num_epochs}\t training loss: {epoch_loss/num_epochs}")
        if epoch_loss/num_classes < pervious_epoch_loss:
            pervious_epoch_loss = epoch_loss/num_classes
            torch.save(model.state_dict(), os.path,join(SVAE_PATH,f'e{epoch}_best.pt'))


# 在之前所有任务上测试
for test_task, test_task_data in task.item():
    test_data = task_data['test']

    test_inputs = tokenizer([data[0] for data in test_data], padding=True, truncation=True, return_tensors='pt')
    test_labels = torch.tensor([label_mapping[data[1]] for data in test_data])

    test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Test on task:\t')
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
    print(":", accuracy)

# 保存模型
