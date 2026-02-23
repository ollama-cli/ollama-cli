from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
import os

class REPL:
    def __init__(self):
        history_file = os.path.expanduser("~/.ollama-cli-history")
        self.session = PromptSession(history=FileHistory(history_file))
        self.style = Style.from_dict({
            'prompt': 'ansicyan bold',
            'command': 'ansigreen',
        })

    def get_input(self) -> str:
        try:
            return self.session.prompt(
                HTML('<prompt><b>&gt; </b></prompt>'),
                style=self.style
            ).strip()
        except KeyboardInterrupt:
            return ""
        except EOFError:
            return "/quit"
