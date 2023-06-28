import pandas as pd
import numpy as np
df = pd.read_csv('ContinualLearning/outputs/albert_vanilla/albert_clinc150_vanilla.csv')
acc = df.iloc[9,1:].mean()
print(acc)
print(df.iloc[1,0])
bt = df.iloc[9,1:].sum() - np.sum([df.iloc[i,i+1] for i in range(10)])
print(bt/9)
with open('ContinualLearning/outputs/albert_vanilla/albert_clinc150_vanilla.txt', 'w') as f:
    f.write(f"acc:{format(acc,'.4f')}\tbt:{format(bt,'.4f')}")