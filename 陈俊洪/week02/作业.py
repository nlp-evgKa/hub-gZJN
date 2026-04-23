import numpy as np
import torch.nn as nn
import torch
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 生成1000个四维数据
X = np.random.rand(1000, 4)

# 生成他们标签
y = np.argmax(X, axis=1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 将numpy数据转换成torch.tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 损失函数(用于分类的交叉熵损失函数)
loss_fn = nn.CrossEntropyLoss()

# 构建模型
class TorchModel(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size: int):
        super().__init__()
        # 定义的两个全连接的线性层
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, matrix:torch.tensor):
        x = self.layer1(matrix)
        y = self.layer2(x)
        return y

def plot(model:nn.Module, X_test:torch.tensor, y_test:torch.tensor):
    """ 绘制混淆矩阵 """
    model.eval()
    with torch.no_grad():
        y_pred_logits = model(X_test)
        y_pred = torch.argmax(y_pred_logits, dim=1)
    
    y_true_np = y_test.numpy()
    y_pred_np = y_pred.numpy()

    cm = confusion_matrix(y_true=y_true_np, y_pred=y_pred_np)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    plt.xlabel('Predict')
    plt.ylabel('true')
    plt.title('Confusion_matrix')
    plt.xticks([0, 1, 2, 3])
    plt.yticks([0, 1, 2, 3])

    plt.show()
def train(X_train:torch.tensor, y_train:torch.tensor, input_size:int, hidden_size:int, output_size:int, epochs: int,
          X_test:torch.tensor, y_test:torch.tensor, batch_size:int=32):
    """ 主训练函数,内部也有模型测试集的验证环节 """
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    # 利用CPU的并行计算能力,将数据分批次处理
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = TorchModel(input_size, hidden_size, output_size)
    model.train()
    
    # 优化器使用SGD 随机梯度下降,(当然也有其他的梯度下降方法,我这里就选了最常见的)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        # 批量训练
        for batch_X, batch_y in train_loader:
            # 清空梯度
            optimizer.zero_grad()
            # 预测数据
            y_pred = model(batch_X)
            # 计算训练损失
            train_loss = loss_fn(y_pred, batch_y)
            # 反向传播
            train_loss.backward()
            # 更新参数
            optimizer.step()
            epoch_train_loss += train_loss.item()

        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader)

        # 在测试集上评估
        with torch.inference_mode():
            model.eval()
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test)
            model.train()

        if epoch%10 == 0:
            # 打印处平均训练损失和测试损失
            print(f"Epoch:{epoch} | train_loss:{avg_train_loss:.5f} | test_loss:{test_loss:.5f}")
    
    return model

def load_and_predict():
    """ 加载并预测 """
    model = TorchModel(input_size=4, hidden_size=10, output_size=4)

    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    return model

def main(argc: int, argv: list[str]) -> int:
    if argc > 1 and argv[1] == "train":
        print("开始训练...")
        model = train(X_train=X_train, y_train=y_train, input_size=4, hidden_size=10,
                     output_size=4, epochs=100, X_test=X_test, y_test=y_test)
        torch.save(model.state_dict(), 'model.pth')
        print("模型已保存到 model.pth")
        return 0

    elif argc > 1 and argv[1] == "predict":
        print("加载模型进行预测...")
        test_sample = torch.tensor([[0.1, 0.9, 0.3, 0.2]], dtype=torch.float32)
        model_load = load_and_predict()
        with torch.no_grad():
            prediction = model_load(test_sample)
            predicted_class = torch.argmax(prediction, dim=1)
            print(f"预测类别: {predicted_class.item()}")
        return 0

    else:
        model = train(X_train=X_train, y_train=y_train, input_size=4, hidden_size=10,
                     output_size=4, epochs=100, X_test=X_test, y_test=y_test)
        torch.save(model.state_dict(), 'model.pth')
        print("模型已保存")

        test_sample = torch.tensor([[0.1, 0.9, 0.3, 0.2]], dtype=torch.float32)
        with torch.no_grad():
            prediction = model(test_sample)
            predicted_class = torch.argmax(prediction, dim=1)
            print(f"预测类别: {predicted_class.item()}")

        plot(model=model, X_test=X_test, y_test=y_test)
        return 0

if __name__ == "__main__":
    argc = len(sys.argv)
    argv = sys.argv
    exit_code = main(argc, argv)
    sys.exit(exit_code)
    
