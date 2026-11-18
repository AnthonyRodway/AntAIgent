"""HTML for Qt's rich-text widgets: Markdown answers, code blocks, and in-app links.

Qt renders a subset of HTML 4 and CSS, so code blocks are single-cell tables (which
reliably take a background colour) and Pygments output uses inline styles.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, quote, urlparse

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name, get_lexer_for_filename
from pygments.util import ClassNotFound


@dataclass(frozen=True)
class Theme:
    dark: bool
    window: str  # behind everything: toolbar, tab bars, gaps between cards
    surface: str  # the cards that hold content
    raised: str  # inputs and buttons
    border: str
    text: str
    muted: str
    accent: str  # links, focus outlines, highlights
    accent_fill: str  # primary buttons; dark enough for white text
    code_bg: str
    bubble: str  # behind your questions in the Ask tab
    line_highlight: str
    pygments_style: str


# Neutral greys in the spirit of Windows 11.
LIGHT = Theme(
    dark=False, window="#f3f3f3", surface="#ffffff", raised="#fbfbfb", border="#e0e0e0",
    text="#1f1f1f", muted="#616161", accent="#0969da", accent_fill="#0969da",
    code_bg="#f6f8fa", bubble="#eef4fd", line_highlight="#fff1b8", pygments_style="friendly",
)
DARK = Theme(
    dark=True, window="#202020", surface="#272727", raised="#2f2f2f", border="#3a3a3a",
    text="#e8e8e8", muted="#a0a0a0", accent="#4493f8", accent_fill="#1f6feb",
    code_bg="#1e1e1e", bubble="#2a3442", line_highlight="#3b3417", pygments_style="github-dark",
)

MONO = "Cascadia Mono, Consolas, 'Courier New', monospace"

# --- Links inside the app ------------------------------------------------------------


def file_link(path: str, start: int | None = None, end: int | None = None) -> str:
    url = "antai:file?p=" + quote(path, safe="")
    if start:
        url += f"&s={start}"
    if end:
        url += f"&e={end}"
    return url


def symbol_link(name: str) -> str:
    return "antai:symbol?n=" + quote(name, safe="")


def parse_link(url: str) -> tuple[str, dict[str, str]]:
    """Return (kind, params) for an antai: link, or ("", {}) for anything else."""
    parsed = urlparse(url)
    if parsed.scheme != "antai":
        return "", {}
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    return parsed.path, params


# --- Code ------------------------------------------------------------------------------


def lexer_for(path: str = "", language: str = ""):
    try:
        if language:
            return get_lexer_by_name(language, stripnl=False)
        if path:
            return get_lexer_for_filename(path, stripnl=False)
    except ClassNotFound:
        pass
    return TextLexer(stripnl=False)


def code_block(code: str, theme: Theme, language: str = "", path: str = "") -> str:
    formatter = HtmlFormatter(noclasses=True, nowrap=True, style=theme.pygments_style)
    body = highlight(code.rstrip("\n"), lexer_for(path, language), formatter)
    return (
        f'<table width="100%" cellspacing="0" cellpadding="8" bgcolor="{theme.code_bg}" '
        f'style="border: 1px solid {theme.border}; margin-top: 4px; margin-bottom: 8px;"><tr><td>'
        f'<pre style="margin: 0; font-family: {MONO};">{body}</pre></td></tr></table>'
    )


# --- Markdown --------------------------------------------------------------------------

_CITATION = re.compile(r"^(?P<path>[^\s:`]+?):(?P<start>\d+)(?:-(?P<end>\d+))?$")


def _markdown(theme: Theme) -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": False, "breaks": False}).enable(["table", "strikethrough"])

    def fence(self, tokens, idx, options, env):
        token = tokens[idx]
        language = token.info.strip().split(" ")[0] if token.info else ""
        return code_block(token.content, theme, language)

    def code_inline(self, tokens, idx, options, env):
        content = tokens[idx].content
        style = f'font-family: {MONO}; background-color: {theme.code_bg};'
        match = _CITATION.match(content.strip())
        paths = env.get("paths") if isinstance(env, dict) else None
        if match and paths is not None and match.group("path") in paths:
            href = file_link(match.group("path"), int(match.group("start")), int(match.group("end") or match.group("start")))
            return f'<a href="{href}" style="color: {theme.accent}; text-decoration: none;"><code style="{style}">{html.escape(content)}</code></a>'
        return f'<code style="{style}">{html.escape(content)}</code>'

    md.add_render_rule("fence", fence)
    md.add_render_rule("code_block", fence)
    md.add_render_rule("code_inline", code_inline)
    return md


_RENDERERS: dict[bool, MarkdownIt] = {}


def markdown_html(text: str, theme: Theme, paths: set[str] | None = None) -> str:
    """Render Markdown; inline `path:start-end` citations of known files become links."""
    md = _RENDERERS.get(theme.dark)
    if md is None:
        md = _RENDERERS[theme.dark] = _markdown(theme)
    out = md.render(text, {"paths": paths} if paths is not None else {})
    if paths:
        out = _link_bracket_citations(out, theme, paths)
    return out.replace("<table>", f'<table border="1" cellspacing="0" cellpadding="4" style="border-color: {theme.border};">')


# gpt-oss sometimes cites in its native style, 【path:start-end】, instead of in backticks.
_BRACKET_CITATION = re.compile(r"【([^【】\s<>]+?):(\d+)(?:-(\d+))?】")


def _link_bracket_citations(html_text: str, theme: Theme, paths: set[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        path = html.unescape(match.group(1))
        if path not in paths:
            return match.group(0)
        start, end = int(match.group(2)), int(match.group(3) or match.group(2))
        href = file_link(path, start, end)
        label = f"{match.group(1)}:{match.group(2)}" + (f"-{match.group(3)}" if match.group(3) else "")
        return f'<a href="{href}" style="color: {theme.accent}; text-decoration: none;">[{label}]</a>'

    return _BRACKET_CITATION.sub(replace, html_text)


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def muted(text: str, theme: Theme) -> str:
    return f'<span style="color: {theme.muted};">{text}</span>'


def link(href: str, label: str, theme: Theme) -> str:
    return f'<a href="{esc(href)}" style="color: {theme.accent}; text-decoration: none;">{esc(label)}</a>'
