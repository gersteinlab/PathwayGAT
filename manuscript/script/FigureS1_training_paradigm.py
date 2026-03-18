import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

f1 = open('../result/JHU_COAD_final_training1.txt', 'r')
lines = f1.readlines()

d_training = {'epoch': [], 'training_loss': [], 'validation_loss': [], 'validation_accuracy': []}
for line in lines:
    epoch = int(line.split(' ')[1].split(',')[0])
    training_loss = float(line.split(' ')[4].split(',')[0])
    validation_loss = float(line.split(' ')[7].split(',')[0])
    validation_accuracy = float(line.split(' ')[9].strip())
    d_training['epoch'].append(epoch)
    d_training['training_loss'].append(training_loss)
    d_training['validation_loss'].append(validation_loss)
    d_training['validation_accuracy'].append(validation_accuracy)
    
df_training = pd.DataFrame(d_training)
fig, ax = plt.subplots(figsize=(6,6), dpi=300)
ax.plot(df_training['epoch'], df_training['training_loss'], label='Training loss')
ax.plot(df_training['epoch'], df_training['validation_loss'], label='Validation loss')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel('Epoch')
ax.legend()
plt.savefig('../plot/JHU_final/training_paradigm/JHU_COAD_training1_loss.pdf')