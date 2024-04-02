import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#卷积核数
#层数 从少到多
#全连接 1-2层
#8 16 8 卷积层输出的特征变少
#残差网络 1-2层
#学习率 影响大
#归一化为（0，1）
#loss函数用原矩阵来算

#TODO:需要一个更加直观的loss函数
#TODO：现在最大的问题是损失不直观：损失函数变小了MUSIC谱就一定会相近吗？

class CustomCNN(nn.Module):
    def __init__(self, M):
        super(CustomCNN, self).__init__()
        self.M = M
        #self.conv1 = nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(M * (M+1), 2 * M * M)
        self.bn1 = nn.BatchNorm1d(2 * M * M)
        #self.conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.fc2 = nn.Linear(2 * M * M, 4 * M * M)
        self.bn2 = nn.BatchNorm1d(4 * M * M)
        #self.conv3 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1)
        self.fc3 = nn.Linear(4 * M * M, 2 * M * M)
        self.bn3 = nn.BatchNorm1d(2 * M * M)
        #self.fc1 = nn.Linear(128 * M * M, 1024)
        #self.fc2 = nn.Linear(1024, 512)
        self.fc0 = nn.Linear(2 * M * M, M * (M+1))
        #self.dropout = nn.Dropout(0.75)

    def forward(self, x):
        #x = x.permute(0, 3, 1, 2)  # Adjust input shape to [N, C, H, W]
        input = x.view(x.size(0), -1)
        conv1_out = torch.tanh(self.bn1(self.fc1(input)))
        conv2_out = torch.tanh(self.bn2(self.fc2(conv1_out)))
        conv3_out = torch.tanh(self.bn3(self.fc3(conv2_out)))
        out = conv1_out + conv3_out
        #out = conv3_out
        out = torch.flatten(out, 1)
        out = self.fc0(out)
        #out = out.view(1, -1)
        out = out.view(-1, 1, int(self.M * (self.M+1)/2), 2)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.fc3(x)
        """
        #x = self.dropout(x)
        #x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        #x = self.fc3(x)
        #x = x.view(-1, self.M, self.M, 2)  # Adjust output shape to [N, H, W, C]
        return out

    def save_parameters(self, file_path):
        torch.save(self.state_dict(), file_path)

    @staticmethod
    def load_parameters(model, file_path):
        model.load_state_dict(torch.load(file_path))
        return model
    
if __name__ == '__main__':
    # 参数和数据加载部分
    M = 32  # 假设输入的高度和宽度为32
    model = CustomCNN(M)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()  # 假设我们的任务是回归，使用MSE作为损失函数

    # 假设的数据，这里只是为了演示
    inputs = torch.randn(10, M, M, 2)  # 生成一些随机数据作为输入
    targets = torch.randn(10, M, M, 2)  # 目标也是随机的，仅作为演示
    
    # 记录损失值的列表
    losses = []

    # 训练过程
    epochs = 5  # 示例使用的epoch数
    for epoch in range(epochs):
        model.train() # 设置模型为训练模式
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 记录当前epoch的损失值
        losses.append(loss.item())
        #print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    #print("Loss after training:", losses[-1])
    # 绘制损失图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('conv_training_loss.png')  # 保存损失图为图片
    #plt.show()  # 显示图表

    # 保存模型参数
    model.save_parameters('custom_cnn.pth')

    # 加载模型参数
    new_model = CustomCNN(M)
    CustomCNN.load_parameters(new_model, 'custom_cnn.pth')

    # 测试步骤（这里简化，只打印损失）
    new_model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        outputs = new_model(inputs)
        loss = criterion(outputs, targets)
        print("Loss after reloading:", loss.item())
