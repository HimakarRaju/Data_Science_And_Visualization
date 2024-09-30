# tic tac toe game

"""

create  a matrix of n,n

create logic
1. check if the game is over
2. check if the game is a draw
3. check if the game is a win for player or AI

Take user inputs in 2 values
val1 , val2

char1 = 'x'
char2 = 'o'

"""
import numpy as np
import os
import random


def user_choice():
    while True:
        char = input("Hi player, Choose 'X' or 'O': ").upper()
        if char in ['X', 'O']:
            return char
        print("Invalid selection. Try again!")


def create_board(n):
    return np.full((n, n), '_')


def print_board(board):
    os.system('cls')
    print(board)


def get_position():
    while True:
        try:
            row = int(input("Enter row (1-3): "))
            col = int(input("Enter column (1-3): "))
            if 1 <= row <= 3 and 1 <= col <= 3:
                return row - 1, col - 1
        except:
            print("Invalid input. Try again!")


def place_char(board, char, row, col):
    if board[row][col] == '_':
        board[row][col] = char
        return True
    return False


def check_win(board, char):
    n = len(board)
    for i in range(n):
        if all(board[i][j] == char for j in range(n)):
            return True
        if all(board[j][i] == char for j in range(n)):
            return True
    if all(board[i][i] == char for i in range(n)):
        return True
    if all(board[i][n-i-1] == char for i in range(n)):
        return True
    return False


def check_draw(board):
    return all(cell != '_' for row in board for cell in row)


def ai_move(board, ai_char):
    while True:
        row = random.randint(0, 2)
        col = random.randint(0, 2)
        if board[row][col] == '_':
            place_char(board, ai_char, row, col)
            return


def play_game():
    board = create_board(3)
    char = user_choice()
    ai_char = 'O' if char == 'X' else 'X'

    while True:
        print_board(board)
        row, col = get_position()
        if place_char(board, char, row, col):
            if check_win(board, char):
                print_board(board)
                print(f"Player {char} wins!")
                break
            elif check_draw(board):
                print_board(board)
                print("It's a draw!")
                break
            ai_move(board, ai_char)
            if check_win(board, ai_char):
                print_board(board)
                print(f"AI {ai_char} wins!")
                break
            elif check_draw(board):
                print_board(board)
                print("It's a draw!")
                break
        else:
            print("Position already occupied. Try again!")


if __name__ == '__main__':
    os.system('cls')
    play_game()
