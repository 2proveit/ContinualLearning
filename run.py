import torch, os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, XLNetForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from datasets import clinc150
from datasets.clinc150 import clinc150_classes
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 初始化BERT模型和分词器
num_classes = 150
model_name = "xlnet-base-cased"
model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=num_classes).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# 定义标签映射
label_mapping = {label: idx for idx, label in enumerate(clinc150_classes)}
# 定义超参数
batch_size = 32
learning_rate = 1e-5
num_epochs = 10
# 保存实验结果
SAVE_PATH = './outputs/xlnet_vanilla'
RESULT_NAME = 'xlnet_clinc150_vanilla'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# 持续学习过程
from datasets.clinc150 import split_data
tasks = split_data
RESULTS = pd.DataFrame(index =["train_"+t for t in tasks.keys()], columns = ["test_"+t for t in tasks.keys()])
# 在当前任务上训练
for task, task_data in tasks.items():
    print("#"*30,"Training on task: ", task,"#"*30)

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
        total_correct = 0
        total_samples = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            with torch.no_grad():
                epoch_loss += loss 
            loss.backward()
            optimizer.step()
            _, predicted_labels = torch.max(outputs.logits, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
        accuracy = total_correct / total_samples

        print(f"task: {task}\t epoch: {epoch}/{num_epochs-1}\t training loss: {epoch_loss/num_epochs}\t accuracy: {accuracy}")
        
        if epoch_loss/num_classes < pervious_epoch_loss:
            pervious_epoch_loss = epoch_loss/num_classes
            torch.save(model.state_dict(), os.path.join(SAVE_PATH,f'e{epoch}_bert.pt'))


    # 在所有任务上测试
    for test_task, test_task_data in tasks.items():
        test_data = test_task_data['test']

        test_inputs = tokenizer([data[0] for data in test_data], padding=True, truncation=True, return_tensors='pt')
        test_labels = torch.tensor([label_mapping[data[1]] for data in test_data])

        test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 在测试集上评估模型性能
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                _, predicted_labels = torch.max(outputs.logits, dim=1)
                total_correct += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f"Test on task:{test_task}\t\t\t accuracy:", accuracy)
        RESULTS[f'test_{test_task}'][f'train_{task}'] = format(accuracy,'.4f')
#写入数据
# 输出结果
RESULTS.to_csv(os.path.join(SAVE_PATH,RESULT_NAME+'.csv'))
Acc = RESULTS.iloc[9,1:].mean() # 任务数量
backward_transfer = RESULTS.iloc[9,1:].sum() - np.sum([RESULTS.iloc[i,i+1] for i in range(10)])
backward_transfer /= 9
with open(os.path.join(SAVE_PATH,RESULT_NAME+'.txt'),'w') as f:
    f.write(f"average accuracy:{Acc,'.4f'}\tbackward transfer:{format(backward_transfer,'4f')}")
print('#'*100)
print(f"average accuracy: {format(Acc,'.4f')}\t backward transfer: {format(backward_transfer,'.4f')}")
print('#'*100)
