#!/usr/bin/env python3
# Authors: jddeetz@gmail.com
"""
This generates training data, which are masked Battleship game boards. 

Generally, there will be three values:
  -1 --> No ship present at this location
   1 --> There is a ship here
   0 --> Fog of war, we do not know whether or not there is a ship at this location
"""

import argparse
import numpy as np
import pickle

BATTLESHIP_SIZES = [5, 4, 3, 3, 2]

def generate_ship_positions() -> np.array:
    """ Generate positions of ships, without overlaps, using brute force. 
        It is assumed that no board is impossible to set all of the ships down.
        Each board is a 10x10 2D grid.

        Returns:
            board: The game board containing ships (1.), and empty space (-1.)
    """
    # Create an empty game board
    board = np.zeros((10,10)) - 1

    # Starting with biggest ship, add it to the board, and move on to smaller ships.
    for ship_size in BATTLESHIP_SIZES:
        ship_not_placed = True
        while ship_not_placed:
            # The ships are always placed from left to right. 
            # To ensure they are randomly placed in left -> right, right-> left, up -> down, and down -> up, 
            # the entire board will be transposed randomly
            if np.random.rand() > 0.5:
                board = board.T

            # Choose a random row between 0 and 10, and column between 0 and 10 - ship_size + 1.
            row_num = np.random.randint(10)
            col_num = np.random.randint(10 - ship_size + 1)

            # Check to see if all spaces from col_num : col_num + ship_size are empty, i.e. equal to -1
            if (board[row_num, col_num:col_num+ship_size] == -1).all():
                # Set these spaces as occupied by the ship
                board[row_num, col_num:col_num+ship_size] = 1
                # Breaks out of while loop
                ship_not_placed = False

    return board


def make_game_boards(num_boards:int, masks_per_board:int, mask_min_fraction:float=0.5) -> None:
    """ Generate game boards for training data.

    Args:
        num_boards: Number of unique positions of ships to generate
        masks_per_board: Number of random masks to generate for
        mask_min_fraction: Generally, games of battleship don't last as long as revealing every single space.
                           This sets a minimum number of spaces that must be masked.
    """
    # Initialize empty arrays for input (foggy) and output (no fog) arrays
    fog_boards = np.empty((num_boards*masks_per_board, 10,10))
    no_fog_boards = np.empty((num_boards*masks_per_board, 10,10))

    for board_num in range(num_boards):
        ### Generate a valid position of battleships
        no_fog_board = generate_ship_positions()

        for mask_num in range(masks_per_board):
            ### Add fog to the game board
            fog_board = no_fog_board.copy()
            # Flatten board to make adding fog more efficient.
            fog_board = fog_board.flatten()
            # Get number of foggy spaces
            num_fog_spaces = np.random.randint(int(100*mask_min_fraction), 100)
            # Get indices for all foggy spaces
            fog_indices = np.random.choice(np.arange(0, 100), num_fog_spaces, replace=False)
            # Set the game board values equal to 0 for all foggy spaces
            fog_board[fog_indices] = 0
            # Reset the board to its original shape
            fog_board = fog_board.reshape(10,10)

            ### Add foggy and non-foggy boards to dataset
            sample_num = board_num * masks_per_board + mask_num
            fog_boards[sample_num, :, :] = fog_board
            no_fog_boards[sample_num, :, :] = no_fog_board

    # Save training data to disk
    pickle.dump(fog_boards, open("data/fog_boards.pkl", 'wb'))
    pickle.dump(no_fog_boards, open("data/no_fog_boards.pkl", 'wb')) 

def main() -> None:
    """ Main function to generate training data for Battleship Game.
    Given the two arguments, the total size of the training data will be num_boards x masks_per_board.

    Example usage of this code is:
    python3 make_training_data.py (--num_boards 1000) (--masks_per_board 10)

    Where:
        num_boards (int): The total number of game boards to generate. The default value is 500.
        masks_per_board (int): The number of times random masks/fog are added to each board. Default is 20.
    """
    # Add and Parse Arguments
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        '--num_boards',
        action="store_const",
        const=True,
        default=500,
        help='The number of unique positions of battleships to generate for the training set.')
    parser.add_argument(
        '--masks_per_board',
        action="store_const",
        const=True,
        default=20,
        help='The number random masks/fog of war to add per game board.')
    args = parser.parse_args()

    # Generate game boards
    make_game_boards(args.num_boards, args.masks_per_board)
            
if __name__ == '__main__':
    main()