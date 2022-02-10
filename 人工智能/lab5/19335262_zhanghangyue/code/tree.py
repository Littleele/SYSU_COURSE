from board import Board

WEIGHT = [[120, -20, 20, 5, 5, 20, -20, 120],
          [-20, -40, -5, -5, -5, -5, -40, -20],
          [20, -5, 15, 3, 3, 15, -5, 20],
          [5, -5, 3, 3, 3, 3, -5, 5],
          [5, -5, 3, 3, 3, 3, -5, 5],
          [20, -5, 15, 3, 3, 15, -5, 20],
          [-20, -40, -5, -5, -5, -5, -40, -20],
          [120, -20, 20, 5, 5, 20, -20, 120]]

move = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]


def validpos(chess, color):
    # 返回当前可以下棋的位置
    poslist = {}  # 用来记录是否有可以落子的位置 并记录其中可以翻转的棋子的坐标
    oppo_color = 1 - color

    for startx in range(8):
        for starty in range(8):
            # 若该位置非空 直接跳过
            if chess.board[startx][starty] != -1:
                continue
            # 遍历周围位置
            for ix, iy in move:
                x = startx + ix
                y = starty + iy
                while chess.onboard(x, y) and chess.board[x][y] == oppo_color:
                    x += ix
                    y += iy
                    # 将夹在自己棋子和对手棋子中间的棋子记录下来
                    if chess.onboard(x, y) and chess.board[x][y] == color:
                        if (startx, starty) not in poslist:
                            poslist[(startx, starty)] = [(startx, starty)]
                        # 存下中间可以翻转的棋子的坐标
                        if ix != 0:
                            m = (x - startx) * ix
                        else:
                            m = (y - starty) * iy
                        for i in range(1, m):
                            poslist[(startx, starty)].append((startx + i * ix, starty + i * iy))
                        break

                    elif chess.onboard(x, y) and chess.board[x][y] == -1:
                        break

    return len(poslist), poslist


# def minimax():

def evaluate(chess, color):
    # 估值函数 为位置权重和稳定动点数
    self_wei = 0
    oppo_wei = 0
    self_val = 0
    oppo_val = 0
    oppo_color = 1 - color
    for i in range(8):
        for j in range(8):
            if chess.board[i][j] == color:
                self_wei += WEIGHT[i][j]
            elif chess.board[i][j] == oppo_color:
                oppo_wei += WEIGHT[i][j]
    # 可落子的数目
    temp_self_val, list = validpos(chess, color)
    temp_oppo_val, list = validpos(chess, oppo_color)
    self_val += temp_self_val
    oppo_val += temp_oppo_val

    # 稳定点数
    steady_self = steadypoint(chess, color)
    steady_oppo = steadypoint(chess, oppo_color)

    return (self_wei - oppo_wei) + 2 * (steady_self - steady_oppo) + 2 * (self_val - oppo_val)


def checkwin(chess):
    white = 0
    black = 0
    for i in range(8):
        for j in range(8):
            if chess.board[i][j] == 1:
                white += 1
            elif chess.board[i][j] == 2:
                black += 1
    if white > black:
        return 1
    else:
        return 0


# 极大节点 ai
def alphabeta(chess, depth, last_alpha, last_beta, color):
    # 叶节点直接返回即可
    if depth == 0:
        score = evaluate(chess, ai_color)
        return score, score, (-1, -1)
    # 初始化节点 alpha beta记录该节点的a和b值
    if color == ai_color:
        alpha = float('-inf')
        beta = last_beta
    else:
        alpha = last_alpha
        beta = float('inf')

    posnum, posdict = validpos(chess, color)
    if posnum == 0:
        color = 1 - color
        posnum, posdict = validpos(chess, color)

        if posnum == 0:  # 棋局结束
            checkwin(chess)
        else:
            return alphabeta(chess, depth - 1, alpha, beta, color)

    validmove = posdict.keys()
    for move in validmove:
        for flipped in posdict[move]:
            chess.board[flipped[0]][flipped[1]] = color
        # 递归遍历
        next_alpha, next_beta, _ = alphabeta(chess, depth - 1, alpha, beta, 1 - color)

        # 撤销刚刚的操作 翻回颜色 取消落子
        for flipped in posdict[move]:
            chess.board[flipped[0]][flipped[1]] = 1 - color
        chess.board[move[0]][move[1]] = -1

        if color == ai_color and alpha < next_beta:
            alpha = next_beta
            pos = (move[0], move[1])
        if color == player_color and beta > next_alpha:
            beta = next_alpha
            pos = (move[0], move[1])
        if beta <= alpha:
            return alpha, beta, pos
    return alpha, beta, pos


def AI_turn(chess, depth_max):
    score, _, pos = alphabeta(chess, depth_max, float('-inf'), float('inf'), ai_color)
    print("电脑落子为: ", pos)
    return pos


def flipped(validdict, color, pos, chess):
    for flipped in validdict[pos]:
        chess.board[flipped[0]][flipped[1]] = color


def steadypoint(chess, color):
    # 计算该颜色不可被翻转的棋子

    total = 0
    oppo_color = 1 - color
    # 首先求出棋盘上该颜色的棋子数
    for i in range(8):
        for j in range(8):
            if chess.board[i][j] == color:
                total += 1
    # 计算对手可翻转的棋子数
    flipped = 0
    _, posdict = validpos(chess, oppo_color)
    validmove = posdict.keys()
    for move in validmove:
        for i in range(1, len(posdict[move])):
            flipped += 1

    return total - flipped


def count_color(chess):
    black = 0
    white = 0
    for i in range(8):
        for j in range(8):
            if chess.board[i][j] == 1:
                black += 1
            elif chess.board[i][j] == 0:
                white += 1

    print("当前棋局: ")
    print("玩家得分: %d" % black)
    print("电脑得分: %d\n" % white)


if __name__ == "__main__":
    # 0 表示该位置空 1 表示该位置为白 2 表示该位置为黑
    chessboard = Board(8)
    chessboard.newboard()
    first_hand = int(input("请选择人先手1 电脑先手0\n"))

    if first_hand:
        print("人先手")
        ai_color = 0
        player_color = 1
        count = 0
        print("玩家棋子为X 电脑棋子为O")
    else:
        print("电脑先手")
        ai_color = 1
        player_color = 0
        count = 1
        print("玩家棋子为O 电脑棋子为X")

    print("初始棋盘")
    chessboard.draw()

    times = 1

    while 1:

        # 人的回合
        if count % 2 == 0:
            if first_hand:
                print("——————————————————————————————————————————————————————————————————————————")
                print("第%d回合:" % times)
            print("玩家落子")
            times += 1
            x, y = [int(x) for x in input('请输入要落子的坐标x,y 空格隔开:\n').split()]
            _, posdict = validpos(chessboard, player_color)
            while (x, y) not in posdict:
                x, y = [int(x) for x in input('该位置不合法 请重新输入\n').split()]
            chessboard.board[x][y] = player_color
            flipped(posdict, player_color, (x, y), chessboard)

            chessboard.draw()
            print("玩家得分为: ", evaluate(chessboard, ai_color))
            print("\n")
        else:
            if first_hand == 0:
                print("——————————————————————————————————————————————————————————————————————————")
                print("第%d回合:" % times)
            print("电脑落子")
            pos = AI_turn(chessboard, 2)
            _, posdict = validpos(chessboard, ai_color)
            flipped(posdict, ai_color, pos, chessboard)
            print("电脑得分为: ", evaluate(chessboard, ai_color))
            chessboard.draw()

        count += 1
