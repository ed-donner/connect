from arena.c4 import make_display
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv(override=True)
    app = make_display()
    app.launch()
