import json
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

# The French state's company directory: free, without a key (https://recherche-entreprises.api.gouv.fr/docs/).
API_URL = "https://recherche-entreprises.api.gouv.fr/search"
MAX_RESULTS = 3
TIMEOUT = 10
STATES = {"A": "active", "C": "closed"}

# Its URL is http://127.0.0.1:8002/mcp.
PORT = 8002
server = MCPServer("sirene")


def search(name: str) -> list[dict]:
    query = urlencode({"q": name, "per_page": MAX_RESULTS})
    try:
        with urlopen(f"{API_URL}?{query}", timeout=TIMEOUT) as response:
            return json.load(response)["results"]
    except (URLError, TimeoutError, ValueError, KeyError) as e:
        raise ToolError(f"The company directory did not answer: {e}") from e


def describe(company: dict) -> str:
    siege = company.get("siege") or {}
    state = STATES.get(company.get("etat_administratif"), "unknown state")
    return (
        f"{company.get('nom_complet')}: SIREN {company.get('siren')}, "
        f"head office SIRET {siege.get('siret')}, {siege.get('adresse')} ({state})"
    )


@server.tool()
def find_siren(name: str) -> str:
    """
    Find the SIREN number of a French company in the official company directory.
    The name is sent to the French state's public API.

    Args:
      name: The name of the company

    Returns:
      The name, SIREN, head office SIRET, address and state of the best matches
    """
    name = name.strip()
    if not name:
        raise ToolError("find_siren: the name is empty")
    companies = search(name)
    if not companies:
        return f"No company found for '{name}'"
    return "\n".join(describe(company) for company in companies[:MAX_RESULTS])


if __name__ == "__main__":
    server.run("streamable-http", port=PORT)
