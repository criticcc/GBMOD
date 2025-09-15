import h5py
import scipy.io
import numpy as np
from scipy.stats import zscore

import scipy.io
import h5py
import numpy as np
from scipy.stats import zscore

import scipy.io
import h5py
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
def load_data(data_path):
    """
    加载 .mat 文件并提取多视图数据和标签，同时对每列进行 Z-score 归一化。

    参数:
        data_path (str): .mat 文件的路径。

    返回:
        v1 (numpy.ndarray): 第一个视图的数据，形状为 (n, d_v1)。
        v2 (numpy.ndarray): 第二个视图的数据，形状为 (n, d_v2)。
        v3 (numpy.ndarray): 第三个视图的数据，形状为 (n, d_v3)。
        label (numpy.ndarray): 标签数据，形状为 (n,)。
    """
    # 首先尝试用 scipy.io.loadmat 加载 MATLAB v7.2 或更低版本的文件
    try:
        data = scipy.io.loadmat(data_path)

        # 提取 X 的第一个元胞
        X_cell = data['X'][0, 0]  # X_cell 是一个形状为 (1, 3) 的数组

        # 提取 X_cell 中的 3 个矩阵并转置
        v1 = X_cell[0, 0].T  # 转置为 (n, d_v1)
        v2 = X_cell[0, 1].T  # 转置为 (n, d_v2)
        v3 = X_cell[0, 2].T  # 转置为 (n, d_v3)

        # 提取 out_label 的第一个元胞
        out_label = data['out_label'][0, 0]  # out_label 是一个形状为 (1, n) 的数组
        label = out_label.flatten()  # 展平为 (n,)


    except (NotImplementedError, ValueError, TypeError):
        # 如果 scipy.io.loadmat 失败，说明是 v7.3 文件，使用 h5py 加载
        print(f"Detected MATLAB v7.3 file, loading with h5py...")
        with h5py.File(data_path, 'r') as f:
            # 调试输出，检查 X 和 out_label 的结构
            print("f['X'] shape:", f['X'].shape, "type:", type(f['X']))
            print("f['X'][0, 0]:", f['X'][0, 0])

            # 获取 X 的引用数组
            X_refs = f['X']  # X 是一个 (5, 1) 的引用数组
            X_cell_refs = f[X_refs[0, 0]][()]  # 解引用 X 的第一个元胞，得到 (3, 1) 引用数组
            print("X_cell_refs shape:", X_cell_refs.shape)

            # 提取 3 个视图的引用并解引用
            X_cell = [f[X_cell_refs[i, 0]][()].T for i in range(3)]  # 从 (3, 1) 数组中取标量引用并转置
            v1, v2, v3 = X_cell[0], X_cell[1], X_cell[2]


            # 获取 out_label 的引用并解引用
            out_label_ref = f['out_label'][0, 0]  # 获取 out_label 的引用
            label = f[out_label_ref][()].flatten()  # 解引用并展平为 (n,)
            # 根据需要调整转置
            if v1.shape[0] != label.shape[0]:  # 如果样本数不在第一个维度
                v1 = v1.T
                v2 = v2.T
                v3 = v3.T
    # ===== Min-Max 归一化（逐列） =====
    scaler = MinMaxScaler()
    v1 = scaler.fit_transform(v1)
    v2 = scaler.fit_transform(v2)
    v3 = scaler.fit_transform(v3)

    # 打印形状以验证
    print("v1 shape:", v1.shape)
    print("v2 shape:", v2.shape)
    print("v3 shape:", v3.shape)
    print("label shape:", label.shape)

    return v1, v2, v3, label
