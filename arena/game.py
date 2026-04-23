from arena.board import Board, RED, YELLOW
from arena.player import Player
from arena.record import get_games, count_games, Result, record_game, ratings
from datetime import datetime
from typing import List
from arena.llm import LLM


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
    def get_games(limit: int = 20) -> List:
        """
        The most recent `limit` games (default 20). Used for the game-history
        table — does not affect ELO, which fetches its own subset.
        """
        return get_games(limit)

    @staticmethod
    def count_games() -> int:
        """Total number of games in the DB (unfiltered, all models)."""
        return count_games()

    @staticmethod
    def get_ratings():
        """
        ELO ratings restricted to games where both players are in the current
        supported-model list. The filter is applied at the DB query layer (see
        record.get_supported_games), so this doesn't scan the whole history.
        """
        return ratings(LLM.all_supported_model_names())

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
