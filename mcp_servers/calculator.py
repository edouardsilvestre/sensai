from mcp.server.mcpserver import MCPServer

server = MCPServer("calculator")


@server.tool()
def add(a: float, b: float) -> float:
    """
    Add two numbers. Use it for any addition instead of computing it yourself.

    Args:
      a: The first number
      b: The second number

    Returns:
      a + b
    """
    return a + b


@server.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers. Use it for any multiplication instead of computing it yourself.

    Args:
      a: The first number
      b: The second number

    Returns:
      a * b
    """
    return a * b


if __name__ == "__main__":
    server.run()
