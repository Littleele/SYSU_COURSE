#!/usr/bin/python
# -*- coding: utf-8 -*-

class Board():
    def __init__(self, n):
        self.n = n
        self.board = self.newboard()
        self.chess = {-1: '.', 0: 'O', 1: 'X'}

    def newboard(self):
        # -1 empty 0 white 1 black
        i = int(self.n / 2)
        board = [[-1] * self.n for _ in range(self.n)]
        board[i][i] = board[i - 1][i - 1] = 0
        board[i][i - 1] = board[i - 1][i] = 1
        return board

    def draw(self):
        index = '0123456789'
        print(' ', *index[:self.n])
        for h, row in zip(index, self.board):
            print(h, *map('OX.'.__getitem__, row))
        print()

    def onboard(self, x, y):
        if x >= 0 and x <= self.n - 1 and y >= 0 and y <= self.n - 1:
            return 1
        else:
            return 0

    # def winnner(self):
