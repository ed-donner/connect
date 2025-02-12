from arena.board import Board, RED, YELLOW
from arena.player import Player
from arena.record import get_games, Result, record_game, ratings
from datetime import datetime
from typing import List


class Game:
    """
    A Game consists of a Board and 2 players
    """

    def __init__(self, model_red: str, model_yellow: str):
        """
        Initialize this Game; a new board, and new Player objects
        """
        self.board = Board()
        self.players = {
            RED: Player(model_red, RED),
            YELLOW: Player(model_yellow, YELLOW),
        }

    def reset(self):
        """
        Restart the game by resetting the board; keep players the same
        """
        self.board = Board()

    def move(self):
        """
        Make the next move. Delegate to the current player to make a move on this board.
        """
        self.players[self.board.player].move(self.board)

    def is_active(self) -> bool:
        """
        Return true if the game hasn't yet ended
        """
        return self.board.is_active()

    def thoughts(self, player) -> str:
        """
        Return the inner thoughts of the given player
        """
        return self.players[player].thoughts()

    @staticmethod
    def get_games() -> List:
        """
        Return all the games stored in the db
        """
        return get_games()

    @staticmethod
    def get_ratings():
        """
        Return the ELO ratings of all players
        """
        return ratings()

    def record(self):
        """
        Store the results of this game in the DB
        """
        red_player = self.players[RED].llm.model_name
        yellow_player = self.players[YELLOW].llm.model_name
        red_won = self.board.winner == RED
        yellow_won = self.board.winner == YELLOW
        result = Result(red_player, yellow_player, red_won, yellow_won, datetime.now())
        record_game(result)

    def run(self):
        """
        If being used outside gradio; move and print in a loop
        """
        while self.is_active():
            self.move()
            print(self.board)
