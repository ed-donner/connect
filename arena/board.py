from arena.board_view import to_svg
from typing import List

RED = 1
YELLOW = -1
EMPTY = 0
show = {EMPTY: "⚪️", RED: "🔴", YELLOW: "🟡"}
pieces = {EMPTY: "", RED: "red", YELLOW: "yellow"}
simple = {EMPTY: "_", RED: "R", YELLOW: "Y"}
cols = "ABCDEFG"


class Board:
    """
    A class to represent a Four-in-the-row Board
    """

    def __init__(self):
        """
        Initialize this instance, starting with empty cells, RED to play
        The latest x,y is used to track the most recent move, so it animates on the display
        """
        self.cells = [[0 for _ in range(7)] for _ in range(6)]
        self.player = RED
        self.winner = EMPTY
        self.draw = False
        self.forfeit = False
        self.latest_x, self.latest_y = -1, -1

    def __repr__(self):
        """
        A visual representation
        """
        result = ""
        for y in range(6):
            for x in range(7):
                result += show[self.cells[5 - y][x]]
            result += "\n"
        result += "\n" + self.message()
        return result

    def message(self):
        """
        A summary of the status
        """
        if self.winner and self.forfeit:
            return f"{show[self.winner]} wins after an illegal move by {show[-1*self.winner]}\n"
        elif self.winner:
            return f"{show[self.winner]} wins\n"
        elif self.draw:
            return "The game is a draw\n"
        else:
            return f"{show[self.player]} to play\n"

    def html(self):
        """
        Return an HTML representation
        """
        result = '<div style="text-align: center;font-size:24px">'
        result += self.__repr__().replace("\n", "<br/>")
        result += "</div>"
        return result

    def svg(self):
        """
        Return an SVG representation
        """
        return to_svg(self)

    def json(self):
        """
        Return a json representation
        """
        result = "{\n"
        result += '    "Column names": ["A", "B", "C", "D", "E", "F", "G"],\n'
        for y in range(6):
            result += f'    "Row {6-y}": ['
            for x in range(7):
                result += f'"{pieces[self.cells[5-y][x]]}", '
            result = result[:-2] + "],\n"
        result = result[:-2] + "\n}"
        return result

    def alternative(self):
        """
        An alternative representation, used in prompting so that the LLM sees this 2 ways
        """
        result = " A B C D E F G\n"
        for y in range(6):
            for x in range(7):
                result += " " + simple[self.cells[5 - y][x]]
            result += "\n"
        return result

    def height(self, x: int) -> int:
        """
        Return the height of the given column
        """
        height = 0
        while height < 6 and self.cells[height][x] != EMPTY:
            height += 1
        return height

    def legal_moves(self) -> List[str]:
        """
        Return the names of columns that are not full
        """
        return [cols[x] for x in range(7) if self.height(x) < 6]

    def illegal_moves(self) -> List[str]:
        """
        Return the names of columns that are full
        """
        return [cols[x] for x in range(7) if self.height(x) == 6]

    def winning_line(self, x: int, y: int, dx: int, dy: int) -> int:
        """
        Return RED or YELLOW if this cell is the start of a 4 in the row going in the direction dx, dy
        Or EMPTY if not
        """
        color = self.cells[y][x]
        for pointer in range(1, 4):
            xp = x + dx * pointer
            yp = y + dy * pointer
            if not (0 <= xp <= 6 and 0 <= yp <= 5) or self.cells[yp][xp] != color:
                return EMPTY
        return color

    def winning_cell(self, x: int, y: int) -> int:
        """
        Return RED or YELLOW if this cell is the start of a 4 in the row
        Or EMPTY if not
        For performance reasons, only look in 4 of the possible 8 directions,
        (because this test will run on both sides of the 4-in-a-row)
        """
        for dx, dy in ((0, 1), (1, 1), (1, 0), (1, -1)):
            if winner := self.winning_line(x, y, dx, dy):
                return winner
        return EMPTY

    def wins(self) -> int:
        """
        Return RED or YELLOW if there is a 4-in-a-row of that color on the board
        Or EMPTY if not
        """
        for y in range(6):
            for x in range(7):
                if winner := self.winning_cell(x, y):
                    return winner
        return EMPTY

    def move(self, x: int):
        """
        Make a move in the given column
        """
        y = self.height(x)
        self.cells[y][x] = self.player
        self.latest_x, self.latest_y = x, y
        if winner := self.wins():
            self.winner = winner
        elif not self.legal_moves:
            self.draw = True
        else:
            self.player = -1 * self.player
        return self

    def is_active(self) -> bool:
        """
        Return true if the game has not yet ended
        """
        return not self.winner and not self.draw
