import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# Load the CSV file into a pandas DataFrame
train_acc = pd.read_excel('train_acc.xlsx')
val_acc = pd.read_excel('validation_acc.xlsx')
train_loss = pd.read_excel('train_loss.xlsx')
val_loss = pd.read_excel('validation_loss.xlsx')
l_rate = pd.read_excel('learningrate.xlsx')

# Create the acc plot using Seaborn
plt.figure(figsize=(10, 6))
train_acc.plot(x='Step', y='Value', kind='line', ax=plt.gca(), label='Training Accuracy')
val_acc.plot(x='Step', y='Value', kind='line', ax=plt.gca(), label='Validation Accuracy')
# mark the 56th epoch of validation acc with a red dot and add to legend
plt.plot(56, val_acc.iloc[56]['Value'], 'ro', label='Highest Validation Accuracy')

plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epochs')

plt.legend(loc='center right')
plt.grid(True)
plt.savefig('acc.png', dpi=300)

# Create the loss plot using Seaborn
plt.figure(figsize=(10, 6))
train_loss.plot(x='Step', y='Value', kind='line', ax=plt.gca(), label='Training Loss')
val_loss.plot(x='Step', y='Value', kind='line', ax=plt.gca(), label='Validation Loss')
# mark the 56th epoch of validation loss with a red dot and add to legend
plt.plot(56, val_loss.iloc[56]['Value'], 'ro', label='Lowest Validation Loss')

plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')

plt.legend(loc='center right')
plt.grid(True)
plt.savefig('loss.png', dpi=300)

# Create the learning rate plot using Seaborn
plt.figure(figsize=(10, 6))
l_rate.plot(x='Step', y='Value', kind='line', ax=plt.gca(), label='Learning Rate')

plt.title('Learning Rate Over Epochs')
plt.xlabel('Epochs')

plt.legend(loc='center right')
plt.grid(True)
plt.savefig('lr.png', dpi=300)

