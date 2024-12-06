import matplotlib.pyplot as plt

# 假设 get_action 返回的 naction 是 shape 为 (num_samples, len_traj_pred, 2) 的数组
naction = to_numpy(get_action(naction))

# 使用 plot_trajs_and_points 绘制轨迹
def plot_naction_trajectory(naction, ax):
    list_trajs = []
    list_points = []
    traj_colors = []
    traj_labels = []
    
    # 遍历所有样本，添加轨迹
    for i in range(naction.shape[0]):  # num_samples
        list_trajs.append(naction[i])  # 轨迹
        list_points.append(naction[i][0])  # 起点
        if i == 0:
            traj_colors.append('cyan')  # 第一条轨迹的颜色
            traj_labels.append('predicted')  # 第一条轨迹标签
        else:
            traj_colors.append('magenta')  # 其他轨迹的颜色
            traj_labels.append(f'sample {i}')  # 其他轨迹标签

    # 绘制轨迹
    plot_trajs_and_points(
        ax=ax,
        list_trajs=list_trajs,  # 所有轨迹
        list_points=list_points,  # 所有起点
        traj_colors=traj_colors,  # 轨迹颜色
        point_colors=['red'] * len(list_points),  # 所有点颜色（起点）
        traj_labels=traj_labels,  # 轨迹标签
        point_labels=['start'] * len(list_points),  # 所有点标签
    )

# 创建一个新的图形和坐标轴来绘制轨迹
fig, ax = plt.subplots()
plot_naction_trajectory(naction, ax)
plt.show()
