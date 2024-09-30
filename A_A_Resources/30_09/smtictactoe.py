import random
import json
import os

# File to store player patterns
PLAYER_HISTORY_FILE = "player_patterns.json"


def load_player_patterns():
    if os.path.exists(PLAYER_HISTORY_FILE):
        with open(PLAYER_HISTORY_FILE, "r") as file:
            return json.load(file)
    return []


def save_player_pattern(pattern):
    history = load_player_patterns()
    history.append(pattern)
    with open(PLAYER_HISTORY_FILE, "w") as file:
        json.dump(history, file)


def print_board(board):
    print("  0   1   2")  # Column indices
    for i, row in enumerate(board):
        print(i, " | ".join(row))  # Row index before each row
        if i < 2:
            print("  " + "-" * 5)  # Divider between rows


def check_winner(board, mark):
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        if all([cell == mark for cell in board[i]]):  # Check rows
            return True
        if all([board[j][i] == mark for j in range(3)]):  # Check columns
            return True
    if all([board[i][i] == mark for i in range(3)]) or all([board[i][2 - i] == mark for i in range(3)]):  # Diagonals
        return True
    return False


def available_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == '-']


def score_board(board):
    if check_winner(board, 'O'):  # AI wins
        return 10
    if check_winner(board, 'X'):  # Player wins
        return -10
    return 0  # Draw or ongoing game


def minimax(board, depth, is_ai_turn, alpha, beta):
    score = score_board(board)
    if score != 0 or not available_moves(board):
        return score

    if is_ai_turn:
        best_score = -float('inf')
        for move in available_moves(board):
            i, j = move
            board[i][j] = 'O'
            best_score = max(best_score, minimax(
                board, depth + 1, False, alpha, beta))
            board[i][j] = '-'  # Undo move
            alpha = max(alpha, best_score)
            if beta <= alpha:  # Alpha-beta pruning
                break
        return best_score
    else:
        best_score = float('inf')
        for move in available_moves(board):
            i, j = move
            board[i][j] = 'X'
            best_score = min(best_score, minimax(
                board, depth + 1, True, alpha, beta))
            board[i][j] = '-'  # Undo move
            beta = min(beta, best_score)
            if beta <= alpha:  # Alpha-beta pruning
                break
        return best_score


def predict_move_from_pattern(board, player_history):
    for pattern in player_history:
        if len(pattern) > 1 and pattern[:-1] == player_history[-(len(pattern) - 1):]:
            last_player_move = pattern[-1]
            if board[last_player_move[0]][last_player_move[1]] == '-':
                print(f"AI predicts player may move to {
                      last_player_move} based on pattern!")
                return last_player_move
    return None


def best_move(board, player_history):
    # Try to predict based on patterns
    predicted_move = predict_move_from_pattern(board, player_history)
    if predicted_move:
        return predicted_move

    # If no prediction is found, use minimax
    best_score = -float('inf')
    best_move = None
    print("\nAI is thinking...")

    for move in available_moves(board):
        i, j = move
        board[i][j] = 'O'
        move_score = minimax(board, 0, False, -float('inf'), float('inf'))
        board[i][j] = '-'  # Undo move

        print(f"AI evaluating move {move}: score {move_score}")

        if move_score > best_score:
            best_score = move_score
            best_move = move

    print(f"\nAI selects move {best_move} with score {best_score}\n")
    return best_move


def tic_tac_toe():
    board = [['-' for _ in range(3)] for _ in range(3)]
    player_history = []
    player_turn = True

    while True:
        print_board(board)

        if player_turn:
            move = input(
                "Enter your move (row and column) as 'row col': ").split()
            i, j = int(move[0]), int(move[1])
            if board[i][j] == '-':
                board[i][j] = 'X'
                player_history.append((i, j))  # Track player move
                if check_winner(board, 'X'):
                    print_board(board)
                    print("You win!")
                    # Store the pattern after the game
                    save_player_pattern(player_history)
                    break
            else:
                print("Invalid move. Try again.")
                continue
        else:
            i, j = best_move(board, player_history)
            board[i][j] = 'O'
            if check_winner(board, 'O'):
                print_board(board)
                print("AI wins!")
                # Store the pattern after the game
                save_player_pattern(player_history)
                break

        if not available_moves(board):
            print_board(board)
            print("It's a tie!")
            # Store the pattern after the game
            save_player_pattern(player_history)
            break

        player_turn = not player_turn


# Start the game
tic_tac_toe()
