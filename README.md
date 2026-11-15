# example
import torch
from torch.utils.data import TensorDataset  # 构造数据集对象
from torch.utils.data import DataLoader  # 数据加载器
from torch import nn
from torch import optim
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_dataset():
    x, y, coef = make_regression(
        n_samples=100,
        n_features=1,
        noise=10,
        coef=True,
        bias=14.5,
        random_state=3
    )
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y, coef


def train(x, y, coef):
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = nn.Linear(in_features=1, out_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs, loss_list, total_loss, total_sample = 100, [], 0.0, 0
    for epoch in range(epochs):
        for train_x, train_y in dataloader:
            y_pred = model(train_x)
            loss = criterion(y_pred, train_y.reshape(-1, 1))
            total_loss += loss.item()
            total_sample += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(total_loss / total_sample)
        print(f'轮数：{epoch + 1}，平均损失值：{total_loss / total_sample}')

    print(f'模型参数,权重:{model.weight},偏置:{model.bias}')
    print(f'{epochs}轮的平均损失分别为：{loss_list}')
    plt.plot(range(epochs),loss_list)
    plt.title('损失值曲线变化图')
    plt.grid()
    plt.show()
    plt.scatter(x,y)
    y_pred = torch.tensor(data = [v*model.weight + model.bias for v in x])
    y_true = torch.tensor(data = [v*coef+14.5 for v in x])
    plt.plot(x,y_pred,color='red',label='预测值')
    plt.plot(x,y_true,color='green',label='真实值')
    plt.legend()
    plt.grid()
    plt.show()
if __name__ == '__main__':
        x, y, coef = create_dataset()
        #print(f'x:{x},y:{y},coef:{coef}')
        train(x, y, coef)
