import numpy as np

def split_train_test(data, labels, test_size=0.2):
    """
    从数据和标签数组中随机抽取测试集，并删除这部分数据之后剩下的作为训练集。
    
    :param data: 完整的数据数组
    :param labels: 对应的标签数组
    :param test_size: 测试集所占的比例
    :return: 训练集数据、训练集标签、测试集数据、测试集标签
    """
    # 确定测试集的大小
    num_samples = data.shape[0]
    num_test_samples = int(num_samples * test_size)
    
    # 随机选择测试集的索引
    test_indices = np.random.choice(num_samples, size=num_test_samples, replace=False)
    
    # 抽取测试集数据和标签
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    
    # 创建训练集的索引（即原始数据中去除测试集索引的部分）
    train_indices = np.delete(np.arange(num_samples), test_indices)
    
    # 抽取训练集数据和标签
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    
    return train_data, train_labels, test_data, test_labels

# 示例使用
# 假设data和labels是你的数据和标签数组
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
labels = np.array([0, 1, 0, 1, 0])

train_data, train_labels, test_data, test_labels = split_train_test(data, labels, test_size=0.4)

print("Train Data:", train_data)
print("Train Labels:", train_labels)
print("Test Data:", test_data)
print("Test Labels:", test_labels)