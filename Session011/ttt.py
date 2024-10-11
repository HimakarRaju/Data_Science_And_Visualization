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
# tic tac toe game

chars = ['X', 'O']
char1, char2 = chars


def User_Choice():
    Choice_Stage = True
    chars = ['X', 'O']
    print('\nHi player, choose a character  \n ')
    Selected_Char = input('Choose  "X" or "O" : ').upper()
    while Choice_Stage:
        if Selected_Char not in chars:
            print("Invalid Selection try again ... \n")
            Selected_Char = input('Choose  "X" or "O" : ').upper()
        else:
            break
    print(f'you chose {Selected_Char}\n')
    return Selected_Char


def starter(n):
    User_Choice()
    num = n  # Number of rows and columns
    game_board = np.full((num, num), '_')  # create a matrix of n,n
    return game_board


def select_position():
    while True:
        try:
            val1 = int(input("Enter the row number (1-3): "))
            val2 = int(input("Enter the column number (1-3): "))
            if 1 <= val1 <= 3 and 1 <= val2 <= 3:
                return val1, val2
        except:
            print("Invalid input. Please enter a number between 1 and 3.")


def place_char():
    game_board = starter()
    row, col = select_position()
    if game_board[row-1][col-1] == '_':
        game_board[row-1][col-1] = select_position()
    else:
        print("Position already occupied. Choose another position.")


def check_win():
    num = 3  # Number of rows and columns
    game_board = starter()
    for i in range(num):
        if game_board[i][0] == game_board[i][1] == game_board[i][2] != '_':
            return True
        if game_board[0][i] == game_board[1][i] == game_board


def main():
    os.system('cls')
    starter(3)


if __name__ == '__main__':
    main()
