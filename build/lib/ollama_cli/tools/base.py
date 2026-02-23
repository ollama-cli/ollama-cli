from typing import Dict, Any, Callable, List, Optional
import json
import inspect

class BaseTool:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    def execute(self, **kwargs) -> Any:
        raise NotImplementedError

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            }
            for t in self.tools.values()
        ]

    def generate_system_prompt_snippet(self) -> str:
        snippet = "Available tools:\n"
        for t in self.tools.values():
            snippet += f"- {t.name}: {t.description}\n"
            snippet += f"  Parameters: {json.dumps(t.parameters)}\n"
        return snippet

# Singleton registry
registry = ToolRegistry()

def tool(name: str, description: str, parameters: Dict[str, Any]):
    """Decorator to register a function as a tool"""
    def decorator(func: Callable):
        class DecoratedTool(BaseTool):
            def __init__(self):
                super().__init__(name, description, parameters)
            def execute(self, **kwargs):
                return func(**kwargs)
        
        registry.register(DecoratedTool())
        return func
    return decorator
