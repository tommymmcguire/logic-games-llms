import re
import matplotlib.pyplot as plt

with open('data/results/trainlog.txt', 'r') as file:
    data = file.read()

# Extract iter and losses using regex pattern
pattern = r'iter (\d+).*loss train: ([\d\.]+), val: ([\d\.n/a]+)'
matches = re.findall(pattern, data)

# Initialize lists to store iter, train, and valid losses
iters, train_losses, val_losses = [], [], []

for match in matches:
    iter_num, train_loss, val_loss = match
    iters.append(int(iter_num))
    train_losses.append(float(train_loss))
    
    # Only add valid (numeric) validation loss values
    if val_loss != 'n/a':
        val_losses.append((int(iter_num), float(val_loss)))
    
# Unzip the validation losses for plotting
val_iters, valid_losses = zip(*val_losses) if val_losses else ([], [])

# Plotting
plt.figure(figsize=(10, 6))

# Plot training loss
plt.plot(iters, train_losses, label='Train Loss', color='blue', marker='.')

# Plot validation loss if available
if valid_losses:
    plt.plot(val_iters, valid_losses, label='Validation Loss', color='red', marker='.')

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Finetuning Mistral7B on Logical Entailment Task')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('data/results/trainlog.png')