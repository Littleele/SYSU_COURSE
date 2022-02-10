import numpy as np
import matplotlib.pyplot as plt
import heapq
import seaborn as sns
import copy
import time

move = [(0, 1), (0, -1), (-1, 0), (1, 0)]


class node:
    def __init__(self, pos, parent=None, step=0, h=0):
        self.pos = pos
        self.step = step
        self.parent = parent
        self.h = h  # 启发函数h(n)

    def __lt__(self, other):  # 重定义'<' 以f(n)=g(n)+h(n)为指标进行队列排序
        return self.step + self.h < other.step + other.h


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        maze = []
        lines = f.readlines()
        for (i, line) in enumerate(lines):
            line = line.strip()
            temp = []
            for (j, data) in enumerate(line):
                # 保存起点和终点信息，用2替换S和E 便于后续画图
                if data == 'S':
                    temp.append(2)
                    s = node([i, j])
                elif data == 'E':
                    temp.append(2)
                    e = node([i, j])
                else:
                    temp.append(int(data))
            maze.append(temp)
    return maze, s, e


def heuristic(start, goal, n):
    """
    启发式函数
    :param start: 当前点位置
    :param goal: 目标点
    :param n: 决定采用何种启发式函数
    :return:
    """
    if n == 1:  # 曼哈顿距离
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    if n == 2:  # 欧式距离
        return np.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)


def Astarsearch(maze, maze2, s, e):
    count = 1  # 记录时间复杂度
    length = 1  # 记录空间复杂度
    row = len(maze)
    col = len(maze[0])
    vis = [[0] * col for i in range(row)]  # 标记该位置是否已访问过 未访问0
    q = []
    maze_2 = copy.deepcopy(maze)
    heapq.heappush(q, s)
    while 1:
        curnode = heapq.heappop(q)

        if curnode.pos == e.pos:
            print('find')
            print('时间复杂度为%d' % count)
            print('空间复杂度为%d' % length)
            print('距离为%d' % curnode.step)
            return curnode

        vis[curnode.pos[0]][curnode.pos[1]] = 1
        maze2[curnode.pos[0]][curnode.pos[1]] = 0.6  # 用于记录探索过的点
        count += 1

        for i in range(4):  # 遍历四个方向移动
            temp_x = curnode.pos[0] + move[i][0]
            temp_y = curnode.pos[1] + move[i][1]

            # 若该点在界内且 有路可走 且此处未被访问过
            if temp_x >= 0 and temp_x < row and temp_y >= 0 and temp_y < col and \
                    maze[temp_x][temp_y] != 1 and vis[temp_x][temp_y] != 1:
                flag = 1
                # 若该点已在边界队列中 但还未拓展 更新其g(n)值 并更新队列的排序
                for NODE in q:
                    if NODE.pos[0] == temp_x and NODE.pos[1] == temp_y and NODE.step > curnode.step + 1:
                        NODE.step = curnode.step + 1
                        NODE.parent = curnode
                        heapq.heapify(q)
                        flag = 0
                        break
                # 若该点不在边界队列中 计算其f值 入队
                if flag:
                    temp_h = heuristic([temp_x, temp_y], e.pos, n=2)
                    temp = node([temp_x, temp_y], curnode, curnode.step + 1, temp_h)
                    heapq.heappush(q, temp)
                    if len(q) > length:
                        length = len(q)  # 更新


# 利用热力图画图
def drawmaze(maze):
    maze2 = np.array(maze)
    sns_plot = sns.heatmap(maze2, cbar=False, cmap="Blues")
    plt.xticks([])
    plt.yticks([])
    plt.title('Maze')
    plt.show()


def drawroute(curnode, maze):
    node = curnode
    while node != None:
        temp_x = node.pos[0]
        temp_y = node.pos[1]
        maze[temp_x][temp_y] = 3
        node = node.parent
    sns_plot = sns.heatmap(maze, cbar=False, cmap="Blues")
    plt.xticks([])
    plt.yticks([])
    plt.title('Path')
    plt.show()


def drawexlored(maze):
    maze2 = np.array(maze)
    sns_plot = sns.heatmap(maze2, cbar=False, cmap="Blues")
    plt.xticks([])
    plt.yticks([])
    plt.title('Explored')
    plt.show()


if __name__ == "__main__":
    start = time.perf_counter()

    maze, s, e = read_file('MazeData.txt')
    maze2 = copy.deepcopy(maze)
    maze3 = copy.deepcopy(maze)

    curnode = Astarsearch(maze, maze2, s, e)
    if curnode is None:
        print("no route")

    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))

    #drawmaze(maze)
    drawexlored(maze2)
    drawroute(curnode, maze3)
