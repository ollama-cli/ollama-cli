import requests
import urllib.parse
from .base import tool

@tool(
    name="web_search",
    description="Search the web for information",
    parameters={
        "query": {"type": "string", "description": "The search query"}
    }
)
def web_search(query: str) -> str:
    """Enhanced web search with DuckDuckGo"""
    try:
        encoded = urllib.parse.quote(query)
        # Use DuckDuckGo Instant Answer API
        api_url = f"https://api.duckduckgo.com/?q={encoded}&format=json"
        response = requests.get(api_url, timeout=10)
        data = response.json()
        
        results = []
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
        
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append(f"- {topic['Text']}")
        
        return "\n".join(results) if results else "No direct results found. Try a different query."
    except Exception as e:
        return f"Search error: {str(e)}"

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
