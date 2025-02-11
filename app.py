import gradio as gr


def greet(name):
    return f"Hello, {name}!"


app = gr.Interface(fn=greet, inputs="text", outputs="text")

if __name__ == "__main__":
    app.launch()
