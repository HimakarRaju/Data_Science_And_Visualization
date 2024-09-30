import numpy as np
import random

# Initialize the Tic Tac Toe board


def initialize_board():
    return np.zeros(9)

# Check if a player has won the game


def check_winner(board, player):
    win_positions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Horizontal wins
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Vertical wins
        [0, 4, 8], [2, 4, 6]              # Diagonal wins
    ]
    for positions in win_positions:
        if all([board[pos] == player for pos in positions]):
            return True
    return False

# Check if the game is a draw


def check_draw(board):
    return not any([spot == 0 for spot in board])

# Get available moves


def get_available_moves(board):
    return np.where(board == 0)[0]


# Q-Learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.995
EXPLORATION_MIN = 0.01

# Q-table for storing learned values
q_table = {}

# Get Q-values for the current board state


def get_q_values(board, player):
    state = tuple(board)  # Convert to tuple to use as a dictionary key
    if state not in q_table:
        q_table[state] = np.zeros(9)  # Initialize Q-values for all moves
    return q_table[state] * player

# Choose the best move using the Q-table or explore


def choose_move(board, player):
    available_moves = get_available_moves(board)
    if np.random.rand() < EXPLORATION_RATE:
        return np.random.choice(available_moves)  # Explore random moves
    q_values = get_q_values(board, player)
    best_move = np.argmax(q_values)  # Exploit best known move
    if best_move in available_moves:
        return best_move
    return np.random.choice(available_moves)

# Update Q-values after each move


def update_q_values(old_board, new_board, player, reward):
    old_state = tuple(old_board)
    new_state = tuple(new_board)
    old_q_values = get_q_values(old_board, player)
    new_q_values = get_q_values(new_board, player)
    old_q_values += LEARNING_RATE * \
        (reward + DISCOUNT_FACTOR * np.max(new_q_values) - old_q_values)

# Play a single game of Tic Tac Toe


def play_game():
    global EXPLORATION_RATE
    board = initialize_board()
    player = 1
    move_history = []

    while True:
        move = choose_move(board, player)
        move_history.append((board.copy(), move, player))

        board[move] = player

        if check_winner(board, player):
            return player, move_history  # Return the winner and history
        if check_draw(board):
            return 0, move_history  # Return draw and history

        player *= -1  # Switch players

# Train the model by playing games


def train_model(episodes=50000):
    global EXPLORATION_RATE
    for episode in range(episodes):
        winner, history = play_game()

        # Update Q-values after the game ends
        # Go through moves in reverse
        for board, move, player in history[::-1]:
            reward = 0
            if winner == player:
                reward = 1  # Reward for winning move
            elif winner != 0:
                reward = -1  # Penalize for losing move
            update_q_values(board, board.copy(), player, reward)

        # Reduce exploration rate
        EXPLORATION_RATE = max(
            EXPLORATION_MIN, EXPLORATION_RATE * EXPLORATION_DECAY)

# Play against the trained AI


def play_against_ai():
    board = initialize_board()
    player = 1  # Player 1 is human, Player -1 is AI
    while True:
        print_board(board)
        if player == 1:
            available_moves = get_available_moves(board)
            move = int(input(f"Choose your move {available_moves}: "))
            if move not in available_moves:
                print("Invalid move. Try again.")
                continue
        else:
            move = choose_move(board, player)
            print(f"AI plays at position {move}")

        board[move] = player
        if check_winner(board, player):
            print_board(board)
            print(f"Player {player} wins!")
            break
        elif check_draw(board):
            print_board(board)
            print("It's a draw!")
            break
        player *= -1  # Switch players

# Print the Tic-Tac-Toe board


def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    for i in range(3):
        print("|".join([symbols[board[j]] for j in range(i * 3, (i + 1) * 3)]))
        if i < 2:
            print("-----")


if __name__ == "__main__":
    print("Training the AI...")
    train_model()
    print("Training complete. Now you can play against the AI.")
    play_against_ai()
