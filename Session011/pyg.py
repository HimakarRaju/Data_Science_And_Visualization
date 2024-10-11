import pygame
import numpy as np
import sys

# Initialize Pygame
pygame.init()

# Screen and grid settings
SCREEN_SIZE = 300
GRID_SIZE = 3
CELL_SIZE = SCREEN_SIZE // GRID_SIZE
LINE_WIDTH = 5
CIRCLE_RADIUS = CELL_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = CELL_SIZE // 4
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Colors for X and O
CROSS_COLOR = (66, 66, 66)
CIRCLE_COLOR = (242, 85, 96)

# Set up the screen
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption('Tic-Tac-Toe')

# Game variables
board = np.full((GRID_SIZE, GRID_SIZE), " ")
current_player = "X"
game_over = False

# Function to draw the grid


def draw_grid():
    screen.fill(WHITE)
    for i in range(1, GRID_SIZE):
        pygame.draw.line(screen, BLACK, (0, CELL_SIZE * i),
                         (SCREEN_SIZE, CELL_SIZE * i), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (CELL_SIZE * i, 0),
                         (CELL_SIZE * i, SCREEN_SIZE), LINE_WIDTH)

# Function to draw X and O


def draw_markers():
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if board[row, col] == 'O':
                pygame.draw.circle(screen, CIRCLE_COLOR, (int(col * CELL_SIZE + CELL_SIZE // 2), int(
                    row * CELL_SIZE + CELL_SIZE // 2)), CIRCLE_RADIUS, CIRCLE_WIDTH)
            elif board[row, col] == 'X':
                pygame.draw.line(screen, CROSS_COLOR, (col * CELL_SIZE + SPACE, row * CELL_SIZE + SPACE),
                                 (col * CELL_SIZE + CELL_SIZE - SPACE, row * CELL_SIZE + CELL_SIZE - SPACE), CROSS_WIDTH)
                pygame.draw.line(screen, CROSS_COLOR, (col * CELL_SIZE + SPACE, row * CELL_SIZE + CELL_SIZE -
                                 SPACE), (col * CELL_SIZE + CELL_SIZE - SPACE, row * CELL_SIZE + SPACE), CROSS_WIDTH)

# Function to check if a player has won


def check_winner(player):
    for row in range(GRID_SIZE):
        if np.all(board[row, :] == player):
            return True
    for col in range(GRID_SIZE):
        if np.all(board[:, col] == player):
            return True
    if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
        return True
    return False

# Function to check if the board is full


def is_board_full():
    return np.all(board != " ")

# Function to restart the game


def restart_game():
    global board, current_player, game_over
    board = np.full((GRID_SIZE, GRID_SIZE), " ")
    current_player = "X"
    game_over = False
    draw_grid()
    draw_markers()


# Main game loop
draw_grid()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Mouse click event
        if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
            mouseX = event.pos[0]  # X position of the click
            mouseY = event.pos[1]  # Y position of the click

            # Calculate row and column based on the mouse click
            clicked_row = mouseY // CELL_SIZE
            clicked_col = mouseX // CELL_SIZE

            # Place marker on the board
            if board[clicked_row, clicked_col] == " ":
                board[clicked_row, clicked_col] = current_player
                draw_markers()

                # Check if the current player won
                if check_winner(current_player):
                    print(f"Player {current_player} wins!")
                    game_over = True
                elif is_board_full():
                    print("It's a tie!")
                    game_over = True
                else:
                    # Switch players
                    current_player = "O" if current_player == "X" else "X"

        # Restart the game when pressing the R key
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                restart_game()

    # Update the display
    pygame.display.update()
