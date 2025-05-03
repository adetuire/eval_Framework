import typer
from .evaluator import run_eval
app = typer.Typer()
@app.command()
def run(config: str):
    """Run evaluation."""
    run_eval(config)
if __name__ == '__main__':
    app()
