from arena.game import Game
from arena.board import RED, YELLOW
from arena.llm import LLM
import gradio as gr
import pandas as pd


css = """
.dataframe-fix .table-wrap {
    min-height: 800px;
    max-height: 800px;
}
footer{display:none !important}
"""

js = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""


def message_html(game) -> str:
    """
    Return the message for the top of the UI
    """
    return (
        f'<div style="text-align: center;font-size:18px">{game.board.message()}</div>'
    )


def format_records_for_table(games):
    """
    Turn the results objects into a pandas DataFrame for the Gradio Dataframe
    """
    df = pd.DataFrame(
        [
            [
                game.when,
                game.red_player,
                game.yellow_player,
                "Red" if game.red_won else "Yellow" if game.yellow_won else "Draw",
            ]
            for game in reversed(games)
        ],
        columns=["When", "Red Player", "Yellow Player", "Winner"],
    )

    # Remove microseconds while preserving datetime format
    df["When"] = pd.to_datetime(df["When"]).dt.floor("s")

    return df


def format_ratings_for_table(ratings):
    """
    Turn the ratings into a List of Lists for the Gradio Dataframe
    """
    items = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    return [[item[0], int(round(item[1]))] for item in items]


def load_callback(red_llm, yellow_llm):
    """
    Callback called when the game is started. Create a new Game object for the state.
    """
    game = Game(red_llm, yellow_llm)
    enabled = gr.Button(interactive=True)
    message = message_html(game)
    return (
        game,
        game.board.svg(),
        message,
        "",
        "",
        enabled,
        enabled,
        enabled,
    )


def leaderboard_callback(game):
    """
    Callback called when the user switches to the Leaderboard tab. Load in the results.
    """
    records_df = format_records_for_table(Game.get_games())
    ratings_df = format_ratings_for_table(Game.get_ratings())
    return records_df, ratings_df


def move_callback(game):
    """
    Callback called when the user clicks to do a single move.
    """
    game.move()
    message = message_html(game)
    if_active = gr.Button(interactive=game.board.is_active())
    return (
        game,
        game.board.svg(),
        message,
        game.thoughts(RED),
        game.thoughts(YELLOW),
        if_active,
        if_active,
    )


def run_callback(game):
    """
    Callback called when the user runs an entire game. Reset the board, run the game, store results.
    Yield interim results so the UI updates.
    """
    enabled = gr.Button(interactive=True)
    disabled = gr.Button(interactive=False)
    game.reset()
    message = message_html(game)
    yield game, game.board.svg(), message, game.thoughts(RED), game.thoughts(
        YELLOW
    ), disabled, disabled, disabled
    while game.board.is_active():
        game.move()
        message = message_html(game)
        yield game, game.board.svg(), message, game.thoughts(RED), game.thoughts(
            YELLOW
        ), disabled, disabled, disabled
    game.record()
    yield game, game.board.svg(), message, game.thoughts(RED), game.thoughts(
        YELLOW
    ), disabled, disabled, enabled


def model_callback(player_name, game, new_model_name):
    """
    Callback when the user changes the model
    """
    player = game.players[player_name]
    player.switch_model(new_model_name)
    return game


def red_model_callback(game, new_model_name):
    """
    Callback when red model is changed
    """
    return model_callback(RED, game, new_model_name)


def yellow_model_callback(game, new_model_name):
    """
    Callback when yellow model is changed
    """
    return model_callback(YELLOW, game, new_model_name)


def player_section(name, default):
    """
    Create the left and right sections of the UI
    """
    all_model_names = LLM.all_model_names()
    with gr.Row():
        gr.HTML(f'<div style="text-align: center;font-size:18px">{name} Player</div>')
    with gr.Row():
        dropdown = gr.Dropdown(
            all_model_names, value=default, label="LLM", interactive=True
        )
    with gr.Row():
        gr.HTML('<div style="text-align: center;font-size:16px">Inner thoughts</div>')
    with gr.Row():
        thoughts = gr.HTML(label="Thoughts")
    return thoughts, dropdown


def make_display():
    """
    The Gradio UI to show the Game, with event handlers
    """
    with gr.Blocks(
        title="C4 Battle",
        css=css,
        js=js,
        theme=gr.themes.Default(primary_hue="sky"),
    ) as blocks:

        game = gr.State()

        with gr.Tabs():
            with gr.TabItem("Game"):

                with gr.Row():
                    gr.HTML(
                        '<div style="text-align: center;font-size:24px">Four-in-a-row LLM Showdown</div>'
                    )
                with gr.Row():
                    with gr.Column(scale=1):
                        red_thoughts, red_dropdown = player_section(
                            "Red", "gpt-4o-mini"
                        )
                    with gr.Column(scale=2):
                        with gr.Row():
                            message = gr.HTML(
                                '<div style="text-align: center;font-size:18px">The Board</div>'
                            )
                        with gr.Row():
                            board_display = gr.HTML()
                        with gr.Row():
                            with gr.Column(scale=1):
                                move_button = gr.Button("Next move")
                            with gr.Column(scale=1):
                                run_button = gr.Button("Run game", variant="primary")
                            with gr.Column(scale=1):
                                reset_button = gr.Button("Start Over", variant="stop")
                        with gr.Row():
                            gr.HTML(
                        '<div style="text-align: center;font-size:16px">See the <a href="https://youtu.be/0OF-ChlKOQY" target="_blank" rel="noopener noreferrer">video walkthrough</a> of the code and <a href="https://github.com/ed-donner/connect" target="_blank" rel="noopener noreferrer">clone</a> the repo</div>'
                    )
                            
                    with gr.Column(scale=1):
                        yellow_thoughts, yellow_dropdown = player_section(
                            "Yellow", "claude-3-5-sonnet-latest"
                        )
            with gr.TabItem("Leaderboard") as leaderboard_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        ratings_df = gr.Dataframe(
                            headers=["Player", "ELO"],
                            label="Ratings",
                            column_widths=[2, 1],
                            wrap=True,
                            col_count=2,
                            row_count=10,
                            max_height=800,
                            elem_classes=["dataframe-fix"],
                        )
                    with gr.Column(scale=2):
                        results_df = gr.Dataframe(
                            headers=["When", "Red Player", "Yellow Player", "Winner"],
                            label="Game History",
                            column_widths=[2, 2, 2, 1],
                            wrap=True,
                            col_count=4,
                            row_count=10,
                            max_height=800,
                            elem_classes=["dataframe-fix"],
                        )
                with gr.Row():
                    gr.HTML(
                        '<div style="text-align: center;font-size:16px">See the <a href="https://youtu.be/0OF-ChlKOQY" target="_blank" rel="noopener noreferrer">video walkthrough</a> of the code and <a href="https://github.com/ed-donner/connect" target="_blank" rel="noopener noreferrer">clone</a> the repo</div>'
                    )

        blocks.load(
            load_callback,
            inputs=[red_dropdown, yellow_dropdown],
            outputs=[
                game,
                board_display,
                message,
                red_thoughts,
                yellow_thoughts,
                move_button,
                run_button,
                reset_button,
            ],
        )
        move_button.click(
            move_callback,
            inputs=[game],
            outputs=[
                game,
                board_display,
                message,
                red_thoughts,
                yellow_thoughts,
                move_button,
                run_button,
            ],
        )
        red_dropdown.change(
            red_model_callback, inputs=[game, red_dropdown], outputs=[game]
        )
        yellow_dropdown.change(
            yellow_model_callback, inputs=[game, yellow_dropdown], outputs=[game]
        )
        run_button.click(
            run_callback,
            inputs=[game],
            outputs=[
                game,
                board_display,
                message,
                red_thoughts,
                yellow_thoughts,
                move_button,
                run_button,
                reset_button,
            ],
        )
        reset_button.click(
            load_callback,
            inputs=[red_dropdown, yellow_dropdown],
            outputs=[
                game,
                board_display,
                message,
                red_thoughts,
                yellow_thoughts,
                move_button,
                run_button,
                reset_button,
            ],
        )

        leaderboard_tab.select(
            leaderboard_callback, inputs=[game], outputs=[results_df, ratings_df]
        )

    return blocks
