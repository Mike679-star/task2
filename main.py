import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from LeNet import LeNet5
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 30
batch = 128
lr = 0.01

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transform,
                               download=True)
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transform,
                              download=True)

train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

model = LeNet5().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss().to(device)


def train(epoch):
    # 三个参数都要初始化为0
    running_loss = 0.0  # 记录总体损失
    running_total = 0  # 记录训练总数
    running_correct = 0  # 记录训练正确数

    for index, batch_data in enumerate(train_dataloader):
        img, label = batch_data
        img = img.cuda()
        label = label.cuda()
        optimizer.zero_grad()

        outputs = model(img)
        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += img.shape[0]
        running_correct += (predicted == label).sum().item()

    print('[%d, %5d]: loss: %.3f, acc: %.2f' % (
        epoch + 1, EPOCH, running_loss / running_total, running_correct / running_total))
    return running_correct / running_total * 100, running_loss / running_total


def test(epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for index, batch_data in enumerate(test_dataloader):
            img, label = batch_data
            img = img.cuda()
            label = label.cuda()
            outputs = model(img)
            _, predicted = torch.max(outputs.data, dim=1)
            total += img.shape[0]
            correct += (predicted == label).sum().item()
    acc = correct / total * 100
    print('[Test: %d / %d]: Accuracy on test set:%.1f' % (epoch + 1, EPOCH, acc))
    return acc


acc_list_train = []
acc_list_test = []
loss_list = []
for i in range(EPOCH):
    model.train()  # 模型训练
    acc_train, loss = train(i)
    acc_list_train.append(acc_train)
    loss_list.append(loss)

    model.eval()  # 模型验证
    acc_test = test(i)
    acc_list_test.append(acc_test)


plt.plot(acc_list_train, label="train")
plt.plot(acc_list_test, label="test")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss On Trainset')
plt.show()
