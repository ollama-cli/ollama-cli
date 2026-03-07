import requests
from .base import tool

@tool(
    name="web_search",
    description="Search the web for information using DuckDuckGo",
    parameters={
        "query": {"type": "string", "description": "The search query"},
        "max_results": {"type": "integer", "description": "Maximum number of results (default: 5)", "optional": True}
    }
)
def web_search(query: str, max_results: int = 5) -> str:
    """Web search using DuckDuckGo via ddgs, with page content for top results"""
    try:
        from ddgs import DDGS
        search_results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                search_results.append(r)

        if not search_results:
            return "No results found. Try a different query."

        output = []
        # Fetch full page content for top 2 results
        for i, r in enumerate(search_results):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            entry = f"**{title}**\n{body}\n{href}"

            if i < 2 and href:
                content = _fetch_page_content(href)
                if content:
                    entry += f"\n\n--- Page Content ---\n{content}"

            output.append(entry)

        return "\n\n".join(output)
    except ImportError:
        return "Error: ddgs not installed. Run: pip install ddgs"
    except Exception as e:
        return f"Search error: {str(e)}"


def _fetch_page_content(url: str, max_chars: int = 3000) -> str:
    """Fetch and extract readable text from a URL."""
    try:
        response = requests.get(url, timeout=8, headers={
            "User-Agent": "Mozilla/5.0 (compatible; ollama-cli/3.0)"
        })
        response.raise_for_status()

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        except ImportError:
            text = response.text

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        return text
    except Exception:
        return ""

@tool(
    name="read_url",
    description="Read and extract text content from a URL",
    parameters={
        "url": {"type": "string", "description": "The URL to read"}
    }
)
def read_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
                
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text[:10000] + ("\n... (truncated)" if len(text) > 10000 else "")
            
        except ImportError:
            # Fallback if bs4 is missing (though we checked)
            return response.text[:10000]
            
    except Exception as e:
        return f"Error reading URL: {str(e)}"
