import matplotlib.pyplot as plt
import numpy as np

def plot_x_marginal():
# === 数据 ===
    x = np.array([1, 2, 3])
    p = np.array([0.25, 0.375, 0.375])

    # === 绘制 PMF ===
    fig, ax = plt.subplots(figsize=(6, 4))
    # ax.stem(x, p, basefmt=" ")
    markerline, stemlines, baseline = ax.stem(x, p, basefmt=" ")
    plt.setp(stemlines, color='black')   # 柱子改为黑色
    plt.setp(markerline, color='black')

    # === 图形样式 ===
    ax.set_xticks(x)
    ax.set_xlabel('X')
    ax.set_ylabel('g(X)')
    ax.set_ylim(0, 1)

    # 在每个柱上标注概率值
    for i, prob in zip(x, p):
        ax.text(i, prob + 0.02, f'{prob:.3f}', ha='center', fontsize=10)

    plt.grid(alpha=0.3, linestyle='--')
    plt.savefig('7/pmf_x_marginal.png')
    plt.show()


def plot_y_marginal():
# === 数据 ===
    x = np.array([0,1])
    p = np.array([0.125, 0.875])

    # === 绘制 PMF ===
    fig, ax = plt.subplots(figsize=(6, 4))
    # ax.stem(x, p, basefmt=" ")

    markerline, stemlines, baseline = ax.stem(x, p, basefmt=" ")
    plt.setp(stemlines, color='black')   # 柱子改为黑色
    plt.setp(markerline, color='black')

    # === 图形样式 ===
    ax.set_xticks(x)
    ax.set_xlabel('Y')
    ax.set_ylabel('h(Y)')
    ax.set_ylim(0, 1)

    # 在每个柱上标注概率值
    for i, prob in zip(x, p):
        ax.text(i, prob + 0.02, f'{prob:.3f}', ha='center', fontsize=10)

    plt.grid(alpha=0.3, linestyle='--')
    plt.savefig('7/pmf_y_marginal.png')
    plt.show()

def plot_joint_distribution():
    # === 定义联合分布 ===
    X = np.array([1, 1, 2, 3])
    Y = np.array([1, 0, 1, 1])
    P = np.array([1 / 8, 1 / 8, 3 / 8, 3 / 8])

    # === 创建3D坐标系 ===
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    # === 绘制3D柱状图 ===
    ax.bar3d(X, Y, np.zeros_like(P),  # x, y, 起始高度
             dx=0.2, dy=0.2, dz=P,  # 柱宽、高
             color='gray', edgecolor='k', alpha=0.8)

    # === 轴标签 ===
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')

    # 调整视角
    ax.view_init(elev=25, azim=-45)
    plt.savefig('7/pmf_joint_dist.png')
    plt.show()

if __name__ == '__main__':
    plot_x_marginal()
    # plot_y_marginal()
    # plot_joint_distribution()