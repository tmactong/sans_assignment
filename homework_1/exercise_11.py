import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Ellipse


def generate_gaussian_samples():
    x,y = np.random.multivariate_normal(
        [1,2],
        [
            [2,1],
            [1,4]
        ],
        size=500
    ).T
    plt.scatter(x,y, s=10, alpha=0.7, edgecolors='none')
    plt.axis('equal')
    # plt.ylim(-0.1, 1.2)
    # plt.yticks([0, 1])
    #plt.xlabel("Time")
    #plt.ylabel("x")
    #plt.savefig('11/multivariate_gaussian_time_series.png')
    plt.show()

def calculate_eigenvalues():
    matrix = np.array(
        [
            [2,1],
            [1,4]
        ]
    )
    eigen_values, eigen_vectors = np.linalg.eig(matrix)
    print(eigen_values)
    print(eigen_vectors)

def plot_rotation():
    phi = np.deg2rad(-22.5)  # 你可以改成任意角度

    # 单位圆上的两个点
    pt1 = (np.cos(phi), np.sin(phi))  # (cos φ, sin φ)
    pt2 = (-np.sin(phi), np.cos(phi))  # (-sin φ, cos φ)

    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=140)

    # 坐标轴
    ax.axhline(0, color='orange', linewidth=1)
    ax.axvline(0, color='orange', linewidth=1)

    # 画完整单位圆（上+下半部分）
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), linestyle='--', color='black')

    # 标出(1,0) 和 (0,1)
    ax.plot([1, 0], [0, 1], 'o', color='deepskyblue')
    ax.text(1.03, -0.03, r'$(1, 0)$', fontsize=11, va='top')
    ax.text(0.02, 1.03, r'$(0, 1)$', fontsize=11, va='bottom')

    # 标出两点
    ax.plot(pt1[0], pt1[1], 'o', color='seagreen')
    ax.text(pt1[0] + 0.03, pt1[1] - 0.02, r'$(\cos\varphi,\ \sin\varphi)$', fontsize=11, ha='left', va='top')

    ax.plot(pt2[0], pt2[1], 'o', color='gold')
    ax.text(pt2[0] - 0.02, pt2[1] + 0.03, r'$(-\sin\varphi,\ \cos\varphi)$', fontsize=11, ha='right', va='bottom')

    # 从原点到两点的箭头
    arrow1 = FancyArrowPatch((0, 0), pt1, arrowstyle='->', mutation_scale=10, linewidth=2, color='blue')
    arrow2 = FancyArrowPatch((0, 0), pt2, arrowstyle='->', mutation_scale=10, linewidth=2, color='blue')
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)

    # 角度弧线 φ（仅在第四象限）
    arc_radius = 0.25
    start_angle = 0
    end_angle = np.rad2deg(phi)  # 负角度
    arc = Arc((0, 0), 2 * arc_radius, 2 * arc_radius, angle=0,
              theta1=start_angle, theta2=end_angle, linewidth=1.5, color='black')
    ax.add_patch(arc)

    # 在弧线附近标 φ
    ax.text(arc_radius * 0.9 * np.cos(phi / 2), arc_radius * 0.9 * np.sin(phi / 2) - 0.05,
            r'22.5', fontsize=12, ha='center')

    # 坐标标签
    ax.text(1.12, 0, r'$x_1$', fontsize=12, va='center', ha='left')
    ax.text(0, 1.12, r'$x_2$', fontsize=12, va='bottom', ha='center')

    # 图形样式设置
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_ellipse_bak():
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create the Ellipse patch
    ellipse = Ellipse(
        (0, 0), 2*np.sqrt(6 + 2*np.sqrt(2)), 2*np.sqrt(6 - 2*np.sqrt(2)), angle=0,
        edgecolor='black', facecolor='none', alpha=0.7
    )
    # Add the ellipse to the axes
    ax.add_patch(ellipse)

    # Set plot limits and aspect ratio for proper visualization
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal', adjustable='box')  # Ensures the ellipse isn't distorted

    ax.set_xlabel('x1')
    ax.set_ylabel('y1')

    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.show()

def plot_standard_ellipse():
    fig, ax = plt.subplots(figsize=(6, 6))

    # 半长轴与半短轴
    a = np.sqrt(6 + 2*np.sqrt(2))
    b = np.sqrt(6 - 2*np.sqrt(2))

    # 椭圆
    ellipse = Ellipse(
        (0, 0), 2*a, 2*b, angle=0,
        edgecolor='black', facecolor='none', alpha=0.7, linewidth=1, linestyle='--'
    )
    ax.add_patch(ellipse)

    # 正方向交点
    x_pos = (a, 0)
    y_pos = (0, b)

    # ---- 椭圆交点标记 ----
    ax.scatter(*x_pos, color='black', s=20, zorder=3)
    ax.scatter(*y_pos, color='black', s=20, zorder=3)

    # ---- 从外部斜着画箭头指向交点 ----
    # 红箭头（右上方向 → x 轴交点）
    ax.annotate(
        '', xy=x_pos, xytext=(a + 0.8, 0.8),
        arrowprops=dict(arrowstyle='->', lw=1, color='black')
    )
    # 蓝箭头（左上方向 → y 轴交点）
    ax.annotate(
        '', xy=y_pos, xytext=(-0.8, b + 0.8),
        arrowprops=dict(arrowstyle='->', lw=1, color='black')
    )

    # ---- 坐标标注 ----
    ax.text(a + 0.1, 1.3, r'($\sqrt{2\lambda_1}$, 0)', color='black',
            fontsize=13, ha='left', va='top')
    ax.text(-0.7, 2.5, r'(0, $\sqrt{2\lambda_2}$)', color='black',
            fontsize=13, ha='right', va='bottom')

    # 坐标轴
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(0, color='gray', linewidth=1)

    # 图形样式
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x₁')
    ax.set_ylabel('y₁')

    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.savefig(f'11/ellipse.png')
    plt.show()

def plot_rotated_ellipse():
    # 半轴长度
    a = np.sqrt(6 + 2 * np.sqrt(2))
    b = np.sqrt(6 - 2 * np.sqrt(2))

    # 旋转角（逆时针）
    theta_deg = 67.5
    theta = np.deg2rad(theta_deg)

    # 长短轴方向单位向量
    u_major = np.array([np.cos(theta), np.sin(theta)])  # 长轴方向
    u_minor = np.array([-np.sin(theta), np.cos(theta)])  # 短轴方向

    # 轴端点
    major_end1 = a * u_major
    major_end2 = -a * u_major
    minor_end1 = b * u_minor
    minor_end2 = -b * u_minor

    fig, ax = plt.subplots(figsize=(6, 6))

    # 椭圆
    ellipse = Ellipse(
        (0, 0), 2 * a, 2 * b, angle=theta_deg,
        edgecolor='black', facecolor='none', alpha=0.7, linewidth=1.2, linestyle='--'
    )
    ax.add_patch(ellipse)

    # 长轴、短轴
    ax.plot([major_end1[0], major_end2[0]], [major_end1[1], major_end2[1]],
            color='gray', linewidth=1)
    ax.plot([minor_end1[0], minor_end2[0]], [minor_end1[1], minor_end2[1]],
            color='gray', linewidth=1)



    # ===== 旋转角 θ 的弧线与箭头 =====
    arc_radius = 1.2
    theta_deg = 67.5  # 逆时针角度
    # 画弧线（从 x 轴正向 0° 到 θ）
    arc = Arc((0, 0), 2 * arc_radius, 2 * arc_radius, angle=0,
              theta1=0, theta2=theta_deg, color='black', lw=1.5)
    ax.add_patch(arc)

    # 在弧线终点放一个顺着弧线方向的箭头
    delta = 3.0  # 取一个很小的角度差（单位：度），用于构造切向箭头
    ang1 = np.deg2rad(theta_deg - delta)
    ang2 = np.deg2rad(theta_deg)

    p1 = (arc_radius * np.cos(ang1), arc_radius * np.sin(ang1))
    p2 = (arc_radius * np.cos(ang2), arc_radius * np.sin(ang2))

    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black",
                        shrinkA=0, shrinkB=0)  # 末端有箭头，贴近弧线
    )

    # 角度文字放在弧线中段位置
    mid = np.deg2rad(theta_deg / 2)
    ax.text(arc_radius * 0.75 * np.cos(mid) + 0.9,
            arc_radius * 0.75 * np.sin(mid) + 0.3,
            r'$\theta = 67.5^\circ$', fontsize=12, ha='center', va='bottom')

    # 坐标轴
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(0, color='gray', linewidth=1)

    # 图形外观
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x₁')
    ax.set_ylabel('y₁')
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.savefig(f'11/rotated_ellipse.png')
    plt.show()

def plot_translated_ellipse():
    # === 椭圆参数 ===
    a = np.sqrt(6 + 2 * np.sqrt(2))  # 半长轴
    b = np.sqrt(6 - 2 * np.sqrt(2))  # 半短轴
    theta_deg = 67.5  # 逆时针旋转角
    theta = np.deg2rad(theta_deg)

    # === 原始中心与新中心 ===
    center_old = np.array([0.0, 0.0])
    center_new = np.array([1.0, 2.0])

    # === 长短轴方向向量 ===
    u_major = np.array([np.cos(theta), np.sin(theta)])  # 长轴方向
    u_minor = np.array([-np.sin(theta), np.cos(theta)])  # 短轴方向

    # === 新中心的长短轴两端点 ===
    major_end1 = center_new + a * u_major
    major_end2 = center_new - a * u_major
    minor_end1 = center_new + b * u_minor
    minor_end2 = center_new - b * u_minor

    fig, ax = plt.subplots(figsize=(6, 6))


    # 新椭圆（实线）
    ellipse_new = Ellipse(
        xy=center_new, width=2 * a, height=2 * b, angle=theta_deg,
        edgecolor='black', facecolor='none', alpha=0.8, linewidth=1, linestyle='--'
    )
    ax.add_patch(ellipse_new)

    # 新椭圆的长轴、短轴
    ax.plot([major_end1[0], major_end2[0]], [major_end1[1], major_end2[1]],
            color='gray', linewidth=1)
    ax.plot([minor_end1[0], minor_end2[0]], [minor_end1[1], minor_end2[1]],
            color='gray', linewidth=1)

    # ===== 平移箭头 =====
    ax.annotate(
        '', xy=center_new, xytext=center_old,
        arrowprops=dict(arrowstyle='->', lw=1, color='black')
    )

    # ===== 中心点标记 =====
    ax.scatter(*center_old, color='gray', s=20, zorder=3)
    ax.text(center_old[0] - 0.3, center_old[1] - 0.3, '(0,0)', color='gray')

    ax.scatter(*center_new, color='black', s=20, zorder=3)
    ax.text(center_new[0] + 0.1, center_new[1] + 0.1, '(1,2)', color='black')

    # 坐标轴
    ax.axhline(0, color='lightgray', linewidth=1)
    ax.axvline(0, color='lightgray', linewidth=1)

    # 图形外观
    ax.set_xlim(-3, 5)
    ax.set_ylim(-2, 6)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x₁')
    ax.set_ylabel('y₁')
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.savefig(f'11/translated_ellipse.png')
    plt.show()


def plot_scatter_and_ellipse():
    a = np.sqrt(6 + 2 * np.sqrt(2))  # 半长轴
    b = np.sqrt(6 - 2 * np.sqrt(2))  # 半短轴
    theta_deg = 67.5  # 逆时针旋转角
    theta = np.deg2rad(theta_deg)
    center_new = np.array([1.0, 2.0])
    u_major = np.array([np.cos(theta), np.sin(theta)])  # 长轴方向
    u_minor = np.array([-np.sin(theta), np.cos(theta)])  # 短轴方向
    major_end1 = center_new + a * u_major
    major_end2 = center_new - a * u_major
    minor_end1 = center_new + b * u_minor
    minor_end2 = center_new - b * u_minor
    x, y = np.random.multivariate_normal(
        [1, 2],
        [[2, 1], [1, 4]],
        size=500
    ).T
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x,y, s=10, alpha=0.7, edgecolors='none')
    ellipse = Ellipse(
        xy=center_new, width=2 * a, height=2 * b, angle=theta_deg,
        edgecolor='black', facecolor='none', alpha=0.9, linewidth=1, linestyle='--'
    )
    ax.add_patch(ellipse)
    ax.plot([major_end1[0], major_end2[0]], [major_end1[1], major_end2[1]],
            color='gray', linewidth=1)
    ax.plot([minor_end1[0], minor_end2[0]], [minor_end1[1], minor_end2[1]],
            color='gray', linewidth=1)
    ax.scatter(*center_new, color='black', s=20, zorder=3)
    ax.text(center_new[0] + 0.1, center_new[1] + 0.1, '(1,2)', color='black')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig('11/scatter.png')
    plt.show()

def multiple_matrix():
    matrix = np.array([
        [0.38268, -0.92388 ],
        [0.92388, 0.38268],
    ]) @ np.array([
        [4.41421, 0],
        [0, 1.58579]
    ]) @ np.array([
        [0.38268, 0.92388],
        [-0.92388, 0.38268],
    ])
    print(matrix)
    matrix_1 = np.array([
        [0.38268, -0.92388 ],
        [0.92388, 0.38268],
    ]) @ np.array([
        [0.22654, 0],
        [0, 0.6306]
    ]) @ np.array([
        [0.38268, 0.92388],
        [-0.92388, 0.38268],
    ])
    print(matrix_1)

def plot_unit_circle():
    fig, ax = plt.subplots(figsize=(6, 6))

    # === 单位圆 ===
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1)

    angle = np.deg2rad(67.5)
    angle_2 = np.deg2rad(67.5 + 90)
    v1 = np.array([[np.cos(angle)], [np.sin(angle)]])
    v2 = np.array([[np.cos(angle_2)], [np.sin(angle_2)]])

    # === 向量箭头（刚好到圆上） ===
    ax.annotate('', xy=v1, xytext=(0,0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black',shrinkA=0, shrinkB=0))
    ax.annotate('', xy=v2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black', shrinkA=0, shrinkB=0))


    # === 转角弧线在单位圆外侧 ===
    arc_radius = 0.5  # 外圈半径
    arc1 = Arc((0,0), 2*arc_radius, 2*arc_radius,
              angle=0, theta1=0, theta2=67.5,
              color='gray', lw=2, linestyle='--')
    arc_radius_1 = 0.75  # 外圈半径
    arc2 = Arc((0, 0), 2 * arc_radius_1, 2 * arc_radius_1,
               angle=0, theta1=0, theta2=67.5+90,
               color='gray', lw=2, linestyle='--')
    ax.add_patch(arc1)
    ax.add_patch(arc2)

    # 弧线箭头（顺时针）
    delta = 2.0
    ang1 = np.deg2rad(67.5 + delta)
    ang2 = np.deg2rad(67.5)
    p1 = arc_radius * np.array([np.cos(ang1), np.sin(ang1)])
    p2 = arc_radius * np.array([np.cos(ang2), np.sin(ang2)])
    ax.annotate('', xy=p1, xytext=p2,
                arrowprops=dict(arrowstyle='-|>', lw=1, color='black', shrinkA=0, shrinkB=0))

    ang3 = np.deg2rad(67.5 + 90 + delta)
    ang4 = np.deg2rad(67.5 + 90)
    p3 = arc_radius_1 * np.array([np.cos(ang3), np.sin(ang3)])
    p4 = arc_radius_1 * np.array([np.cos(ang4), np.sin(ang4)])
    ax.annotate('', xy=p3, xytext=p4,
                arrowprops=dict(arrowstyle='-|>', lw=1, color='black', shrinkA=0, shrinkB=0))

    # === 角度文字（靠更外侧一些） ===
    mid = np.deg2rad(67.5 / 2)
    ax.text(0.9 * np.cos(mid), 0.9 * np.sin(mid),
            r'$\frac{3}{8}\pi=67.5^\circ$', fontsize=11, ha='center', va='top')

    mid_1 = np.deg2rad(90+ 67.5 / 2)
    ax.text(0.9 * np.cos(mid_1), 0.9 * np.sin(mid_1),
            r'$\frac{7}{8}\pi=157.5^\circ$', fontsize=11, ha='center', va='top')

    # === 坐标标注 ===
    ax.text(np.cos(angle)-0.2, np.sin(angle)+0.1, r'($cos\frac{3}{8}\pi$, $sin\frac{3}{8}\pi$)', color='black', fontsize=11, va='bottom')
    ax.text(np.cos(angle_2)-1, np.sin(angle_2)+0.2, r'($cos\frac{7}{8}\pi$, $sin\frac{7}{8}\pi$)',
            color='black', fontsize=11, ha='left', va='top')

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
    plt.savefig('11/unit_circle.png')
    plt.show()

def main():
    # generate_gaussian_samples()
    # calculate_eigenvalues()
    # plot_rotation()
    # plot_standard_ellipse()
    # plot_rotated_ellipse()
    # plot_translated_ellipse()
    # plot_scatter_and_ellipse()
    # multiple_matrix()
    plot_unit_circle()


if __name__ == '__main__':
    main()