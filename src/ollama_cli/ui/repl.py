from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory, History
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
import os


_CHECKPOINT_SENTINEL = "__checkpoint_restore__"


class CheckpointRestore(Exception):
    """Raised by get_input() when the user double-taps ESC."""
    pass


class _FilteredHistory(FileHistory):
    """FileHistory that filters out sentinel values."""

    def store_string(self, string: str) -> None:
        if string.strip() == _CHECKPOINT_SENTINEL:
            return
        super().store_string(string)


class REPL:
    def __init__(self):
        history_file = os.path.expanduser("~/.ollama-cli-history")

        # Key bindings
        self.bindings = KeyBindings()

        @self.bindings.add('escape', 'escape')
        def _double_escape(event):
            """Double-tap ESC: request checkpoint restore."""
            event.current_buffer.reset()
            event.app.exit(result=_CHECKPOINT_SENTINEL)

        self.session = PromptSession(
            history=_FilteredHistory(history_file),
            enable_suspend=True,
            key_bindings=self.bindings,
        )
        self.style = Style.from_dict({
            'prompt': 'ansicyan bold',
            'command': 'ansigreen',
        })

    def get_input(self) -> str:
        try:
            result = self.session.prompt(
                HTML('<prompt><b>&gt; </b></prompt>'),
                style=self.style
            ).strip()
            if result == _CHECKPOINT_SENTINEL:
                raise CheckpointRestore()
            return result
        except KeyboardInterrupt:
            return ""
        except EOFError:
            return "/quit"
