from arena.llm import LLM
from arena.board import pieces, cols
from pydantic import BaseModel, Field, create_model
from typing import Literal, Type
import logging


class Player:
    """
    One AI player. Owns the prompts, the LLM, and a dynamic Pydantic schema
    whose `move_column` is a Literal over the currently-legal columns — so the
    structured-output layer makes illegal moves unrepresentable.
    """

    def __init__(self, model: str, color: int):
        self.color = color
        self.model = model
        self.llm = LLM.create(self.model)
        self.evaluation = ""
        self.threats = ""
        self.opportunities = ""
        self.strategy = ""

    def system(self) -> str:
        return f"""You are playing the board game Connect 4.
Players take turns to drop counters into one of 7 columns A, B, C, D, E, F, G.
The winner is the first player to get 4 counters in a row in any direction (horizontal, vertical, or diagonal).
You are {pieces[self.color]} and your opponent is {pieces[self.color * -1]}.

Look first for a move that wins immediately, then for one that blocks an opponent's win next move; avoid any move that hands your opponent an immediate winning reply.
Beyond those, prefer moves that create double threats — two winning paths the opponent cannot block in a single turn.
Otherwise build central control and develop connections. When the right move is evident, commit; the strongest play is often the obvious one.
Avoid over-thinking. Do not reconsider moves multiple times. After considering opportunities, threats, and the overall position, pick your best move and stick to it.
"""

    def user(self, board) -> str:
        return f"""It is your turn to make a move as {pieces[self.color]}.

Current board (R = red, Y = yellow, . = empty):

{board.display()}

{board.next_drops()}

Pick the column you want to drop your counter into."""

    def move_schema(self, legal_moves: list[str]) -> Type[BaseModel]:
        """Build a Pydantic schema with move_column constrained to legal letters."""
        move_literal = Literal[tuple(legal_moves)]  # type: ignore[valid-type]
        return create_model(
            "Move",
            evaluation=(str, Field(description="My assessment of the board")),
            threats=(str, Field(description="Threats from my opponent that I should block")),
            opportunities=(str, Field(description="My best chances to win")),
            strategy=(str, Field(description="My thought process for this move")),
            move_column=(move_literal, Field(description="The column to drop my counter into")),
        )

    def process_move(self, result: BaseModel | None, board):
        """
        Apply the move. `result` is a parsed schema instance or None (retries
        exhausted). None forfeits. Otherwise move_column is guaranteed legal
        by the schema; we just make the move and cache the thoughts.
        """
        if result is None:
            logging.error(f"{self.model} forfeited: LLM call failed after retries")
            board.forfeit = True
            board.winner = -1 * board.player
            return
        col = cols.find(result.move_column)
        board.move(col)
        self.evaluation = result.evaluation
        self.threats = result.threats
        self.opportunities = result.opportunities
        self.strategy = result.strategy

    def move(self, board):
        legal_moves = board.legal_moves()
        schema = self.move_schema(legal_moves)
        result = self.llm.send(self.system(), self.user(board), schema)
        self.process_move(result, board)

    def thoughts(self):
        result = '<div style="text-align: left;font-size:14px"><br/>'
        result += f"<b>Evaluation:</b><br/>{self.evaluation}<br/><br/>"
        result += f"<b>Threats:</b><br/>{self.threats}<br/><br/>"
        result += f"<b>Opportunities:</b><br/>{self.opportunities}<br/><br/>"
        result += f"<b>Strategy:</b><br/>{self.strategy}"
        result += "</div>"
        return result

    def switch_model(self, new_model_name: str):
        self.llm = LLM.create(new_model_name)
