import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

matrix_a = np.array([
    [4, 3, 2, 1],
    [1, 2, 1, 3],
    [3, 1, 0, 2],
    [2, 3, 4, 5]
])

matrix_b = np.array([
    [5, 0, 2, -1],
    [0, 2, 0, -1],
    [2, 0, 1, 0],
    [-1, -1, 0, 2]
])

matrix_c = np.array([
    [2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2]
])

matrix_d = np.array([
    [4, 3],
    [3, 1]
])

matrix_e = np.array([
    [4, 3, 2, 1],
    [1, 2, 1, 3],
    [3, 1, 0, 2],
    [2, 3, 4, 5],
    [2, 1, 3, 2]
])

v_e = np.array([
    [1], [2], [1], [5], [5]
])

v_f = np.array([
    [1], [2], [1], [5]
])

matrix_f = matrix_b

b1 = np.array([[1], [0]])
b2 = np.array([[0], [1]])
b3 = np.array([[-1], [0]])
b4 = np.array([[0], [-1]])

def diagonalize_matrix(m):
    eigen_values, eigen_vectors = np.linalg.eig(m)
    print(f'eigen values', eigen_values)
    print(f'eigen vectors',eigen_vectors)
    inverse = np.linalg.inv(eigen_vectors)
    diagonal_matrix = np.diag(eigen_values)
    print(f'inverse of eigenvectors is {inverse}')
    print(f'D @ P^-1 @ b_1: {diagonal_matrix @ inverse @ b2}')
    # print(f'pdp^-1 is {eigen_vectors @ diagonal_matrix @ inverse}')


def projection_matrix(m, v):
    P = m @ np.linalg.inv(m.T @ m) @ m.T
    print(P)
    print(f'{20 * '#'}')
    print(m.T @ (v - P @ v))
    print(f'{20 * '#'}')
    print(P @ v)




def plot_rotated_vector_angle_outside():
    fig, ax = plt.subplots(figsize=(6, 6))

    # === 单位圆 ===
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1)

    # 原始向量 (1,0)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    v3 = np.array([0.0, -1.0])
    v4 = np.array([-1.0, 0.0])

    # 顺时针旋转 31.7°（负角）
    angle_deg = -31.7
    angle_rad = np.deg2rad(angle_deg)
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    v1_1 = R @ v1  # 单位长度
    v2_1 = R @ v2
    v3_1 = R @ v3
    v4_1 = R @ v4

    # === 向量箭头（刚好到圆上） ===
    ax.annotate('', xy=v1, xytext=(0,0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black',shrinkA=0, shrinkB=0))
    ax.annotate('', xy=v1_1, xytext=(0,0),
                arrowprops=dict(arrowstyle='->', lw=1, color='blue',shrinkA=0, shrinkB=0))

    ax.annotate('', xy=v2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black', shrinkA=0, shrinkB=0))
    ax.annotate('', xy=v2_1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='blue', shrinkA=0, shrinkB=0))

    ax.annotate('', xy=v3, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black', shrinkA=0, shrinkB=0))
    ax.annotate('', xy=v3_1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='blue', shrinkA=0, shrinkB=0))

    ax.annotate('', xy=v4, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black', shrinkA=0, shrinkB=0))
    ax.annotate('', xy=v4_1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='blue', shrinkA=0, shrinkB=0))

    # === 转角弧线在单位圆外侧 ===
    arc_radius = 1.25  # 外圈半径
    arc1 = Arc((0,0), 2*arc_radius, 2*arc_radius,
              angle=0, theta1=angle_deg, theta2=0,
              color='black', lw=1)
    arc2 = Arc((0, 0), 2 * arc_radius, 2 * arc_radius,
               angle=-90, theta1=angle_deg, theta2=0,
               color='black', lw=1)
    arc3 = Arc((0, 0), 2 * arc_radius, 2 * arc_radius,
               angle=-180, theta1=angle_deg, theta2=0,
               color='black', lw=1)
    arc4 = Arc((0, 0), 2 * arc_radius, 2 * arc_radius,
               angle=-270, theta1=angle_deg, theta2=0,
               color='black', lw=1)
    ax.add_patch(arc1)
    ax.add_patch(arc2)
    ax.add_patch(arc3)
    ax.add_patch(arc4)

    # 弧线箭头（顺时针）
    delta = 2.0
    ang1 = np.deg2rad(angle_deg + delta)
    ang2 = np.deg2rad(angle_deg)
    p1 = arc_radius * np.array([np.cos(ang1), np.sin(ang1)])
    p2 = arc_radius * np.array([np.cos(ang2), np.sin(ang2)])
    ax.annotate('', xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle='-|>', lw=1, color='black', shrinkA=0, shrinkB=0))

    ang3 = np.deg2rad(angle_deg -90 + delta)
    ang4 = np.deg2rad(angle_deg -90)
    p3 = arc_radius * np.array([np.cos(ang3), np.sin(ang3)])
    p4 = arc_radius * np.array([np.cos(ang4), np.sin(ang4)])
    ax.annotate('', xy=p4, xytext=p3,
                arrowprops=dict(arrowstyle='-|>', lw=1, color='black', shrinkA=0, shrinkB=0))

    ang5 = np.deg2rad(angle_deg-180 + delta)
    ang6 = np.deg2rad(angle_deg-180)
    p5 = arc_radius * np.array([np.cos(ang5), np.sin(ang5)])
    p6 = arc_radius * np.array([np.cos(ang6), np.sin(ang6)])
    ax.annotate('', xy=p6, xytext=p5,
                arrowprops=dict(arrowstyle='-|>', lw=1, color='black', shrinkA=0, shrinkB=0))

    ang7 = np.deg2rad(angle_deg-270 + delta)
    ang8 = np.deg2rad(angle_deg-270)
    p7 = arc_radius * np.array([np.cos(ang7), np.sin(ang7)])
    p8 = arc_radius * np.array([np.cos(ang8), np.sin(ang8)])
    ax.annotate('', xy=p8, xytext=p7,
                arrowprops=dict(arrowstyle='-|>', lw=1, color='black', shrinkA=0, shrinkB=0))

    # === 角度文字（靠更外侧一些） ===
    mid = np.deg2rad(angle_deg/2)
    ax.text(1.55*np.cos(mid), 1.55*np.sin(mid),   # 调整半径为 1.55
            r'$31.7^\circ$', fontsize=11, ha='center', va='top')

    mid_1 = np.deg2rad(angle_deg / 2 -90)
    ax.text(1.55 * np.cos(mid_1) , 1.55 * np.sin(mid_1) + 0.2,  # 调整半径为 1.55
            r'$31.7^\circ$', fontsize=11, ha='center', va='top')

    mid_2 = np.deg2rad(angle_deg / 2 -180)
    ax.text(1.55 * np.cos(mid_2)+0.1, 1.55 * np.sin(mid_2) ,  # 调整半径为 1.55
            r'$31.7^\circ$', fontsize=11, ha='center', va='top')

    mid_3 = np.deg2rad(angle_deg / 2 -270)
    ax.text(1.55 * np.cos(mid_3) + 0.1, 1.55 * np.sin(mid_3) -0.1 ,  # 调整半径为 1.55
            r'$31.7^\circ$', fontsize=11, ha='center', va='top')

    # === 坐标标注 ===
    #ax.text(1.1, 0, '(1, 0)', color='black', fontsize=11, va='bottom')
    #ax.text(v2[0]+0.08, v2[1]-0.08, f'({v2[0]:.2f}, {v2[1]:.2f})',
    #        color='black', fontsize=11, ha='left', va='top')

    # === 坐标轴与样式 ===
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.grid(alpha=0.4, linestyle='--')
    plt.legend(loc='upper right')
    plt.savefig('2/rotation.png')
    plt.show()


def plot_expansion():
    fig, ax = plt.subplots(figsize=(6, 6))

    # === 单位圆 ===
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1)

    # 原始向量 (1,0)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    v3 = np.array([0.0, -1.0])
    v4 = np.array([-1.0, 0.0])

    # 顺时针旋转 31.7°（负角）
    angle_deg = -31.7
    angle_rad = np.deg2rad(angle_deg)
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    v1_1 = R @ v1  # 单位长度
    v2_1 = R @ v2
    v3_1 = R @ v3
    v4_1 = R @ v4

    D = np.array([[5.854, 0], [0, -0.854]])
    v1_2 = D @ v1_1
    v2_2 = D @ v2_1
    v3_2 = D @ v3_1
    v4_2 = D @ v4_1


    # === 向量箭头 ===
    ax.annotate('', xy=v1_1, xytext=(0,0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black',shrinkA=0, shrinkB=0))

    ax.annotate('', xy=v2_1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black', shrinkA=0, shrinkB=0))

    ax.annotate('', xy=v3_1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black', shrinkA=0, shrinkB=0))

    ax.annotate('', xy=v4_1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black', shrinkA=0, shrinkB=0))

    ax.annotate('', xy=v1_2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='blue', shrinkA=0, shrinkB=0))

    ax.annotate('', xy=v2_2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='blue', shrinkA=0, shrinkB=0))

    ax.annotate('', xy=v3_2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='blue', shrinkA=0, shrinkB=0))

    ax.annotate('', xy=v4_2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='blue', shrinkA=0, shrinkB=0))

    ax.annotate('', xy=v1_2, xytext=v1_1,
                arrowprops=dict(arrowstyle='->', lw=1, color='gray', shrinkA=0, shrinkB=0, linestyle='dotted'))

    ax.annotate('', xy=v2_2, xytext=v2_1,
                arrowprops=dict(arrowstyle='->', lw=1, color='gray', shrinkA=0, shrinkB=0, linestyle='dotted'))

    ax.annotate('', xy=v3_2, xytext=v3_1,
                arrowprops=dict(arrowstyle='->', lw=1, color='gray', shrinkA=0, shrinkB=0, linestyle='dotted'))

    ax.annotate('', xy=v4_2, xytext=v4_1,
                arrowprops=dict(arrowstyle='->', lw=1, color='gray', shrinkA=0, shrinkB=0, linestyle='dotted'))

    # === 坐标标注 ===
    #ax.text(1.1, 0, '(1, 0)', color='black', fontsize=11, va='bottom')
    #ax.text(v2[0]+0.08, v2[1]-0.08, f'({v2[0]:.2f}, {v2[1]:.2f})',
    #        color='black', fontsize=11, ha='left', va='top')

    # === 坐标轴与样式 ===
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.grid(alpha=0.4, linestyle='--')
    plt.legend(loc='upper right')
    plt.savefig('2/expansion.png')
    plt.show()

def plot_second_rotation():
    fig, ax = plt.subplots(figsize=(6, 6))

    # === 单位圆 ===
    theta = np.linspace(0, 2*np.pi, 400)
    # ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1)
    ax.plot(5 * np.cos(theta), 5 * np.sin(theta), 'k--', lw=1)
    ax.plot(np.sqrt(10) * np.cos(theta), np.sqrt(10) * np.sin(theta), 'k--', lw=1)

    # 原始向量 (1,0)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    v3 = np.array([0.0, -1.0])
    v4 = np.array([-1.0, 0.0])

    # 顺时针旋转 31.7°（负角）
    angle_deg = -31.7
    angle_rad = np.deg2rad(angle_deg)
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    v1_1 = R @ v1  # 单位长度
    v2_1 = R @ v2
    v3_1 = R @ v3
    v4_1 = R @ v4

    D = np.array([[5.854, 0], [0, -0.854]])
    v1_2 = D @ v1_1
    v2_2 = D @ v2_1
    v3_2 = D @ v3_1
    v4_2 = D @ v4_1

    angle = 31.7
    angle_nrad = np.deg2rad(angle)
    P = np.array([[np.cos(angle_nrad), -np.sin(angle_nrad)],
                  [np.sin(angle_nrad),  np.cos(angle_nrad)]])

    v1_3 = P @ v1_2
    v2_3 = P @ v2_2
    v3_3 = P @ v3_2
    v4_3 = P @ v4_2

    # === 向量箭头（刚好到圆上） ===
    ax.annotate('', xy=v1_2, xytext=(0,0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black',shrinkA=0, shrinkB=0))
    ax.annotate('', xy=v1_3, xytext=(0,0),
                arrowprops=dict(arrowstyle='->', lw=1, color='blue',shrinkA=0, shrinkB=0))

    ax.annotate('', xy=v2_2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black', shrinkA=0, shrinkB=0))
    ax.annotate('', xy=v2_3, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='blue', shrinkA=0, shrinkB=0))

    ax.annotate('', xy=v3_2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black', shrinkA=0, shrinkB=0))
    ax.annotate('', xy=v3_3, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='blue', shrinkA=0, shrinkB=0))

    ax.annotate('', xy=v4_2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black', shrinkA=0, shrinkB=0))
    ax.annotate('', xy=v4_3, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='blue', shrinkA=0, shrinkB=0))


    # === 坐标标注 ===
    #ax.text(1.1, 0, '(1, 0)', color='black', fontsize=11, va='bottom')
    #ax.text(v2[0]+0.08, v2[1]-0.08, f'({v2[0]:.2f}, {v2[1]:.2f})',
    #        color='black', fontsize=11, ha='left', va='top')

    # === 转角弧线在单位圆外侧 ===
    arc_radius_1 = np.sqrt(10) + 0.25  # 外圈半径
    arc1 = Arc((0, 0), 2 * arc_radius_1, 2 * arc_radius_1,
               angle=-13.2825, theta1=0, theta2=31.7,
               color='black', lw=1)
    arc2 = Arc((0, 0), 2 * arc_radius_1, 2 * arc_radius_1,
               angle=180-13.2825, theta1=0, theta2=31.7,
               color='black', lw=1)

    arc_radius_2 = 5.25
    arc3 = Arc((0, 0), 2 * arc_radius_2, 2 * arc_radius_2,
               angle=5.1524, theta1=0, theta2=31.7,
               color='black', lw=1)
    arc4 = Arc((0, 0), 2 * arc_radius_2, 2 * arc_radius_2,
               angle=-180+5.1524, theta1=0, theta2=31.7,
               color='black', lw=1)
    ax.add_patch(arc1)
    ax.add_patch(arc2)
    ax.add_patch(arc3)
    ax.add_patch(arc4)

    # 弧线箭头（顺时针）
    delta = 2.0
    ang1 = np.deg2rad(-13.2825 + 31.7 + delta)
    ang2 = np.deg2rad(-13.2825 + 31.7)
    p1 = arc_radius_1 * np.array([np.cos(ang1), np.sin(ang1)])
    p2 = arc_radius_1 * np.array([np.cos(ang2), np.sin(ang2)])
    ax.annotate('', xy=p1, xytext=p2,
                arrowprops=dict(arrowstyle='-|>', lw=1, color='black', shrinkA=0, shrinkB=0))

    ang3 = np.deg2rad(-180 - 13.2825 + 31.7 + delta)
    ang4 = np.deg2rad(-180 - 13.2825 + 31.7)
    p3 = arc_radius_1 * np.array([np.cos(ang3), np.sin(ang3)])
    p4 = arc_radius_1 * np.array([np.cos(ang4), np.sin(ang4)])
    ax.annotate('', xy=p3, xytext=p4,
                arrowprops=dict(arrowstyle='-|>', lw=1, color='black', shrinkA=0, shrinkB=0))

    ang5 = np.deg2rad(5.1524+ 31.7 + delta)
    ang6 = np.deg2rad(5.1524 + 31.7)
    p5 = arc_radius_2 * np.array([np.cos(ang5), np.sin(ang5)])
    p6 = arc_radius_2 * np.array([np.cos(ang6), np.sin(ang6)])
    ax.annotate('', xy=p5, xytext=p6,
                arrowprops=dict(arrowstyle='-|>', lw=1, color='black', shrinkA=0, shrinkB=0))

    ang7 = np.deg2rad(-180 + 5.1524 + 31.7 + delta)
    ang8 = np.deg2rad(-180 + 5.1524 + 31.7)
    p7 = arc_radius_2 * np.array([np.cos(ang7), np.sin(ang7)])
    p8 = arc_radius_2 * np.array([np.cos(ang8), np.sin(ang8)])
    ax.annotate('', xy=p7, xytext=p8,
                arrowprops=dict(arrowstyle='-|>', lw=1, color='black', shrinkA=0, shrinkB=0))

    # === 角度文字（靠更外侧一些） ===
    mid = np.deg2rad((5.1524 + 31.7) / 2)
    ax.text(5.55 * np.cos(mid) +0.2, 5.55 * np.sin(mid)+0.5,
            r'$31.7^\circ$', fontsize=11, ha='center', va='top')

    mid_1 = np.deg2rad((5.1524 + 31.7) / 2 - 180)
    ax.text(5.55 * np.cos(mid_1), 5.55 * np.sin(mid_1),
            r'$31.7^\circ$', fontsize=11, ha='center', va='top')

    mid_2 = np.deg2rad((-13.2825 + 31.7)/2)
    ax.text((np.sqrt(10)+0.55) * np.cos(mid_2) + 0.4, (np.sqrt(10)+0.55) * np.sin(mid_2) - 0.7 ,
            r'$31.7^\circ$', fontsize=11, ha='center', va='top')

    mid_3 = np.deg2rad((-13.2825 + 31.7)/2 - 180)
    ax.text((np.sqrt(10)+0.55) * np.cos(mid_3) - 0.2, (np.sqrt(10)+0.55) * np.sin(mid_3) - 0.1,  # 调整半径为 1.55
            r'$31.7^\circ$', fontsize=11, ha='center', va='top')

    # === 坐标轴与样式 ===
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.grid(alpha=0.4, linestyle='--')
    plt.legend(loc='upper right')
    plt.savefig('2/second_rotation.png')
    plt.show()



if __name__ == '__main__':
    # diagonalize_matrix(matrix_d)
    # projection_matrix(matrix_f, v_f)
    # plot_rotated_vector_angle_outside()
    # plot_expansion()
    plot_second_rotation()
