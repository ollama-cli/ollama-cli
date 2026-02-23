from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.syntax import Syntax
from typing import Optional

console = Console()

def print_markdown(text: str):
    console.print(Markdown(text))

def print_panel(text: str, title: Optional[str] = None, style: str = "blue"):
    console.print(Panel(text, title=title, border_style=style))

def print_error(text: str):
    console.print(f"[bold red]Error:[/bold red] {text}")

def print_status(text: str):
    console.print(f"[bold blue]>>>[/bold blue] {text}")

class StreamingDisplay:
    def __init__(self, title: Optional[str] = None):
        self.text = ""
        self.live = Live(Text(self.text), refresh_per_second=10)
        self.title = title

    def __enter__(self):
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.live.stop()

    def update(self, chunk: str):
        self.text += chunk
        # Try to parse markdown if it looks complete-ish, otherwise just show text
        # For performance, we might just show raw text during streaming
        self.live.update(Markdown(self.text) if "```" in self.text else Text(self.text))
