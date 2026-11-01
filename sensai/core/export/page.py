import html

from sensai.core.export.markdown import blocks, inline

NOTICE = "Made with Sensai. The answers come from a language model: check every claim against its source."
_LISTS = {"bullet": "ul", "numbered": "ol"}
# One file with no script and nothing to load: a shared page reaches no other server, not even to tell it was read.
_PAGE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<meta name="referrer" content="no-referrer">
<title>{title}</title>
<style>{style}</style>
</head>
<body>
<main>
<h1>{title}</h1>
<p class="details">{details}</p>
{parts}
<p class="notice">{notice}</p>
</main>
</body>
</html>
"""
_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #faf9f5; --bubble: #efece3; --text: #2b2a27; --muted: #76746c; --border: #e2dfd4; --accent: #c96442;
  --code: #f0ede4;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #262624; --bubble: #1a1918; --text: #ece9e1; --muted: #a3a097; --border: #3e3d39; --accent: #d9774f;
    --code: #1f1e1d;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 16px/1.6 Georgia, "Times New Roman", serif; }
main { max-width: 46rem; margin: 0 auto; padding: 40px 16px; overflow-wrap: anywhere; }
h1 { margin: 0 0 4px; font-size: 1.7rem; font-weight: 400; line-height: 1.25; }
h2, h3, h4, h5, h6 { margin: 1.3em 0 .4em; font-weight: 600; line-height: 1.3; }
h2 { font-size: 1.25rem; } h3 { font-size: 1.1rem; } h4, h5, h6 { font-size: 1rem; }
p { margin: .6em 0; }
.details, .sources, .notice, .author { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.details, .notice { color: var(--muted); font-size: .85rem; }
.notice { margin-top: 40px; padding-top: 12px; border-top: 1px solid var(--border); }
.part { margin-top: 24px; }
.author { margin: 0 0 4px; color: var(--accent); font-size: .8rem; font-weight: 600; }
.you .text { padding: 8px 16px; border-radius: 16px; background: var(--bubble); }
.sources { margin: 8px 0 0; padding: 0; color: var(--muted); font-size: .8rem; list-style: none; }
code { padding: 1px 4px; border-radius: 4px; background: var(--code); font: .88em ui-monospace, Consolas, Menlo, monospace; }
pre { padding: 12px; border-radius: 8px; background: var(--code); overflow-x: auto; }
pre code { padding: 0; background: none; }
@media print { main { max-width: none; padding: 0; } }
"""


def _inline(text: str) -> str:
    return inline(text, ("<strong>", "</strong>"), ("<em>", "</em>"), ("<code>", "</code>"))


def body(text: str, heading_shift: int = 0) -> str:
    """The Markdown as HTML. Everything is escaped: what the model writes can come from a leaked document."""
    out: list[str] = []
    opened: str | None = None
    for block in blocks(text):
        tag = _LISTS.get(block.kind)
        if opened != tag:
            if opened:
                out.append(f"</{opened}>")
            if tag:
                out.append(f"<{tag}>")
            opened = tag
        if block.kind == "heading":
            level = min(block.number + heading_shift, 6)
            out.append(f"<h{level}>{_inline(block.text)}</h{level}>")
        elif block.kind == "paragraph":
            out.append("<p>" + "<br>".join(_inline(line) for line in block.text.split("\n")) + "</p>")
        elif block.kind == "bullet":
            out.append(f"<li>{_inline(block.text)}</li>")
        elif block.kind == "numbered":
            out.append(f'<li value="{block.number}">{_inline(block.text)}</li>')
        else:
            out.append(f"<pre><code>{html.escape(block.text, quote=False)}</code></pre>")
    if opened:
        out.append(f"</{opened}>")
    return "\n".join(out)


def _part(author: str, text: str, sources: list[tuple[int, str]], you: str) -> str:
    # Under an author, the headings of the text come below the one of the page.
    label = f'<p class="author">{html.escape(author)}</p>' if author else ""
    cited = "".join(f"<li>[{number}] {html.escape(citation)}</li>" for number, citation in sources)
    listed = f'<ul class="sources">{cited}</ul>' if cited else ""
    kind = "part you" if author == you else "part"
    return f'<section class="{kind}">{label}<div class="text">{body(text, 2 if author else 0)}</div>{listed}</section>'


def page(title: str, details: str, parts: list[tuple[str, str, list[tuple[int, str]]]], you: str) -> str:
    return _PAGE.format(
        title=html.escape(title), details=html.escape(details), style=_STYLE, notice=html.escape(NOTICE),
        parts="\n".join(_part(author, text, sources, you) for author, text, sources in parts),
    )
