"""Reusable UI primitives for the Streamlit product demo."""

from __future__ import annotations

import base64
import hashlib
import html
import textwrap
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.app.branding import LOGO_PATH, PRODUCT_NAME, SHORT_NAME, TAGLINE


STYLE_PATH = Path(__file__).resolve().parents[1] / "styles.css"

NODE_COLORS = {
    "artist": "#2f4a46",
    "gallery": "#7a5b2f",
    "museum": "#3f654f",
    "collector": "#6d5a78",
    "curator": "#8a6d32",
    "exhibition": "#68717a",
    "acquisition": "#536f7a",
    "auction_result": "#6a665e",
    "press_mentions": "#8a6d32",
}


def load_design_system() -> None:
    """Inject the app CSS once per render."""
    st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_html(markup: str) -> None:
    """Render custom HTML without Markdown treating indentation as code."""
    normalized = textwrap.dedent(markup).strip()
    compact = "".join(line.strip() for line in normalized.splitlines() if line.strip())
    st.markdown(compact, unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    """Render a polished product brand block in the sidebar."""
    mark = _logo_markup()
    title = "" if _logo_data_uri() else f'<div class="ani-brand-title">{escape(PRODUCT_NAME)}</div>'
    markup = textwrap.dedent(
        f"""
        <div class="ani-sidebar-brand">
          <div class="ani-brand-stack">
            {mark}
            {title}
            <div class="ani-brand-subtitle">{escape(TAGLINE)}</div>
          </div>
        </div>
        """
    ).strip()
    compact = "".join(line.strip() for line in markup.splitlines() if line.strip())
    st.sidebar.markdown(compact, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, refresh_timestamp: str) -> None:
    """Render the top product header."""
    logo = _logo_markup("header")
    kicker = "" if _logo_data_uri() else f'<div class="ani-page-kicker">{escape(PRODUCT_NAME)}</div>'
    render_html(
        f"""
        <div class="ani-top-header">
          <div class="ani-header-brand">
            {logo}
            <div>
              {kicker}
              <div class="ani-page-title">{escape(title)}</div>
              <div class="ani-page-subtitle">{escape(subtitle)}</div>
            </div>
          </div>
          <div class="ani-refresh">Last refresh<br>{escape(refresh_timestamp)}</div>
        </div>
        """
    )


def metric_card(label: str, value: str, caption: str = "", tone: str = "neutral") -> None:
    """Render a compact KPI card."""
    render_html(
        f"""
        <div class="ani-card ani-score-card">
          <div class="ani-card-caption">{escape(label)}</div>
          <div class="ani-metric-value">{escape(value)}</div>
          <div class="ani-metric-caption">{escape(caption)}</div>
          <div style="margin-top: .65rem;">{pill(tone.title(), tone=tone)}</div>
        </div>
        """
    )


def score_card(label: str, value: str, caption: str = "", confidence: str = "", tone: str = "neutral") -> None:
    """Render a high-emphasis score card for model outputs."""
    badge = confidence_badge(confidence, tone) if confidence else ""
    render_html(
        f"""
        <div class="ani-score-panel">
          <div class="ani-score-topline">
            <span>{escape(label)}</span>
            {badge}
          </div>
          <div class="ani-score-value">{escape(value)}</div>
          <div class="ani-score-caption">{escape(caption)}</div>
        </div>
        """
    )


def evidence_card(title: str, value: str, body: str, tone: str = "neutral") -> None:
    """Render one evidence card."""
    render_html(
        f"""
        <div class="ani-evidence-card {escape(tone)}">
          <div class="ani-evidence-card-head">
            <span>{escape(title)}</span>
            {pill(value, tone=tone)}
          </div>
          <div class="ani-evidence-card-body">{escape(body)}</div>
        </div>
        """
    )


def evidence_grid(rows: pd.DataFrame, limit: int = 6) -> None:
    """Render evidence rows as a responsive card grid."""
    if rows.empty:
        empty_state("No evidence available", "Add sourced events or relationships to populate this section.")
        return
    cards = []
    for _, row in rows.head(limit).iterrows():
        tone = "success" if str(row.get("evidence", "")).lower().startswith(("museum", "major")) else "neutral"
        if "confidence" in str(row.get("evidence", "")).lower():
            tone = "warning" if str(row.get("value", "")).lower() in {"low", "medium"} else "success"
        cards.append(
            f"""
            <div class="ani-evidence-card {tone}">
              <div class="ani-evidence-card-head">
                <span>{escape(row.get("evidence", ""))}</span>
                {pill(row.get("value", ""), tone=tone)}
              </div>
              <div class="ani-evidence-card-body">{escape(row.get("why_it_matters", ""))}</div>
            </div>
            """
        )
    render_html(f'<div class="ani-card-grid">{"".join(cards)}</div>')


def confidence_badge(label: str, tone: str = "neutral") -> str:
    """Return a compact confidence badge."""
    safe_tone = tone if tone in {"neutral", "success", "warning", "danger"} else "neutral"
    return f'<span class="ani-confidence-badge {safe_tone}">{escape(label)}</span>'


def relationship_cards(rows: pd.DataFrame, limit: int = 6) -> None:
    """Render relationship/timeline rows as readable cards."""
    if rows.empty:
        empty_state("No relationship evidence", "No dated relationships are available for the selected filters.")
        return
    cards = []
    for _, row in rows.head(limit).iterrows():
        detail = str(row.get("detail", "") or "")
        detail_markup = _relationship_detail_markup(detail)
        title = str(row.get("counterparty", "") or "")
        subtitle = str(row.get("counterparty_type", "") or "")
        subtitle_markup = f'<div class="ani-relationship-meta">{escape(subtitle)}</div>' if subtitle and subtitle != title else ""
        cards.append(
            f"""
            <div class="ani-relationship-card">
              <div class="ani-relationship-topline">
                <div class="ani-relationship-date">{escape(row.get("date", ""))}</div>
                <div class="ani-relationship-type">{escape(row.get("relationship", ""))}</div>
              </div>
              <div class="ani-relationship-main">{escape(title)}</div>
              {subtitle_markup}
              {detail_markup}
            </div>
            """
        )
    render_html(f'<div class="ani-relationship-list">{"".join(cards)}</div>')


def _relationship_detail_markup(detail: str) -> str:
    """Render pipe-separated evidence metadata as readable key-value snippets."""
    if not detail:
        return ""
    items = []
    for part in (piece.strip() for piece in detail.split("|")):
        if not part:
            continue
        if ":" in part:
            label, value = part.split(":", 1)
            items.append(
                f"""
                <span class="ani-relationship-detail">
                  <span>{escape(label.strip())}</span>
                  {escape(value.strip())}
                </span>
                """
            )
        else:
            items.append(f'<span class="ani-relationship-detail">{escape(part)}</span>')
    if not items:
        return ""
    return f'<div class="ani-relationship-details">{"".join(items)}</div>'


def empty_state(title: str, body: str = "") -> None:
    """Render a quiet empty state."""
    render_html(
        f"""
        <div class="ani-empty-state">
          <div class="ani-empty-title">{escape(title)}</div>
          <div class="ani-empty-body">{escape(body)}</div>
        </div>
        """
    )


def section_heading(title: str, caption: str = "") -> None:
    """Render a reusable section heading."""
    render_html(
        f"""
        <div class="ani-section-heading">
          <h2>{escape(title)}</h2>
          <span>{escape(caption)}</span>
        </div>
        """
    )


def pill(label: str, tone: str = "neutral") -> str:
    """Return HTML for a status pill."""
    safe_tone = tone if tone in {"neutral", "success", "warning", "danger"} else "neutral"
    return f'<span class="ani-pill {safe_tone}">{escape(label)}</span>'


def render_pill(label: str, tone: str = "neutral") -> None:
    """Render a status pill."""
    render_html(pill(label, tone))


def note_box(text: str, warning: bool = False) -> None:
    """Render a low-noise product note or warning."""
    class_name = "ani-warning-box" if warning else "ani-note-box"
    render_html(f'<div class="{class_name}">{escape(text)}</div>')


def driver_list(items: Iterable[str], empty_text: str = "No material signals available.") -> None:
    """Render a list of explanation drivers."""
    rows = list(items)
    if not rows:
        rows = [empty_text]
    markup = "".join(f'<div class="ani-driver-item">{escape(str(item))}</div>' for item in rows)
    render_html(f'<div class="ani-driver-list">{markup}</div>')


def table_label(label: str) -> None:
    """Render a small table label."""
    render_html(f'<div class="ani-table-label">{escape(label)}</div>')


def _logo_markup(placement: str = "sidebar") -> str:
    """Return the configured PNG logo with a text fallback."""
    source = _logo_data_uri()
    if not source:
        return f'<div class="ani-logo-mark" aria-label="{escape(PRODUCT_NAME)}">{escape(SHORT_NAME)}</div>'
    class_name = "ani-logo-image header" if placement == "header" else "ani-logo-image sidebar"
    return f'<img class="{class_name}" src="{source}" alt="{escape(PRODUCT_NAME)} logo" />'


def _logo_data_uri() -> str:
    """Return the logo PNG as an embeddable data URI."""
    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def display_identifier(value: object) -> str:
    """Return readable text for an internal identifier."""
    text = str(value or "").strip()
    if not text:
        return ""
    if "://" in text or "/" in text:
        return text
    base = text.removesuffix(".csv")
    parts = [part for part in base.split("_") if part]
    if len(parts) <= 1:
        return _title_identifier_text(base)
    prefixes = {"artist", "gallery", "museum", "collector", "curator", "institution"}
    event_prefixes = {
        "auction": "Auction result",
        "press": "Press mention",
        "event": "Event",
        "fair": "Art fair",
        "acq": "Acquisition",
        "rel": "Relationship",
    }
    prefix = parts[0].lower()
    if prefix in prefixes:
        return _title_identifier_text(" ".join(parts[1:]))
    if prefix in event_prefixes:
        detail_parts = parts[1:]
        detail_parts = [part for part in detail_parts if part.lower() not in prefixes]
        detail = _title_identifier_text(" ".join(detail_parts))
        return f"{event_prefixes[prefix]} - {detail}" if detail else event_prefixes[prefix]
    return _title_identifier_text(" ".join(parts))


def _title_identifier_text(text: str) -> str:
    """Title-case identifier text while preserving common abbreviations."""
    abbreviations = {"id": "ID", "url": "URL", "usd": "USD", "1y": "1Y", "3y": "3Y"}
    return " ".join(abbreviations.get(part.lower(), part.capitalize()) for part in text.split())


def render_interactive_graph(
    graph: nx.MultiDiGraph,
    selected_node: str | None = None,
    max_edges: int = 140,
    height: int = 650,
) -> None:
    """Render a lightweight SVG graph with pan, zoom, labels, and tooltips."""
    if graph.number_of_nodes() == 0:
        note_box("No graph data matches the current filters.")
        return

    visible_edges = list(graph.edges(data=True))[:max_edges]
    visible_nodes = set(graph.nodes())
    if visible_edges:
        visible_nodes = {source for source, _, _ in visible_edges} | {target for _, target, _ in visible_edges}

    visible_graph = nx.Graph()
    visible_graph.add_nodes_from(visible_nodes)
    visible_graph.add_edges_from((source, target) for source, target, _ in visible_edges)
    positions = nx.spring_layout(visible_graph, seed=11, k=1.1)

    width = 980
    svg_height = max(420, height - 90)
    normalized = _normalize_positions(positions, width, svg_height)
    graph_id = "ani_graph_" + hashlib.md5("|".join(sorted(visible_nodes)).encode("utf-8")).hexdigest()[:10]

    edge_markup = []
    for source, target, edge_data in visible_edges:
        x1, y1 = normalized[source]
        x2, y2 = normalized[target]
        relationship = display_identifier(edge_data.get("relationship_type", "relationship"))
        confidence = float(edge_data.get("confidence_score", 0.0) or 0.0)
        edge_markup.append(
            f"""
            <line class="ani-graph-edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"
              stroke-width="{1.0 + confidence:.2f}">
              <title>{escape(source)} -> {escape(target)} | {escape(relationship)} | confidence {confidence:.2f}</title>
            </line>
            """
        )

    node_markup = []
    for node_id in visible_nodes:
        x, y = normalized[node_id]
        data = graph.nodes[node_id]
        node_type = str(data.get("node_type", "unknown"))
        name = str(data.get("name") or data.get("title") or display_identifier(node_id))
        color = NODE_COLORS.get(node_type, "#8f8a80")
        radius = 11 if node_id != selected_node else 16
        stroke = "#181716" if node_id == selected_node else "#ffffff"
        label_y = y + radius + 15
        node_markup.append(
            f"""
            <g class="ani-graph-node">
              <circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" stroke="{stroke}" stroke-width="2">
                <title>{escape(name)} | {escape(display_identifier(node_type))} | {escape(node_id)}</title>
              </circle>
              <text x="{x:.1f}" y="{label_y:.1f}" text-anchor="middle">{escape(_short_label(name))}</text>
            </g>
            """
        )

    graph_styles = """
    <style>
      :root {
        --ani-graph-surface: #ffffff;
        --ani-graph-surface-alt: #fbfaf7;
        --ani-graph-ink: #181716;
        --ani-graph-muted: #6f6a61;
        --ani-graph-border: #ded8cf;
        --ani-graph-border-strong: #c9c0b4;
      }
      @media (prefers-color-scheme: dark) {
        :root {
          --ani-graph-surface: #181715;
          --ani-graph-surface-alt: #1f1d1a;
          --ani-graph-ink: #f3efe7;
          --ani-graph-muted: #b8afa3;
          --ani-graph-border: #39342e;
          --ani-graph-border-strong: #4b453d;
        }
      }
      body { margin: 0; background: transparent; }
      .ani-graph-shell {
        background: var(--ani-graph-surface);
        border: 1px solid var(--ani-graph-border);
        border-radius: 8px;
        box-shadow: 0 14px 36px rgba(24, 23, 22, 0.08);
        overflow: hidden;
      }
      .ani-graph-toolbar {
        align-items: center;
        border-bottom: 1px solid var(--ani-graph-border);
        color: var(--ani-graph-muted);
        display: flex;
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 11px;
        justify-content: space-between;
        letter-spacing: .06em;
        padding: 10px 13px;
        text-transform: uppercase;
      }
      .ani-graph-svg {
        background: var(--ani-graph-surface-alt);
        display: block;
        height: 560px;
        width: 100%;
      }
      .ani-graph-edge {
        stroke: var(--ani-graph-border-strong);
        stroke-opacity: .72;
      }
      .ani-graph-node text {
        fill: var(--ani-graph-ink);
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 11px;
        font-weight: 600;
        pointer-events: none;
      }
    </style>
    """

    html_doc = f"""
    {graph_styles}
    <div class="ani-graph-shell">
      <div class="ani-graph-toolbar">
        <span>Scroll to zoom. Drag to pan.</span>
        <span>{len(visible_nodes)} nodes / {len(visible_edges)} relationships</span>
      </div>
      <svg id="{graph_id}" class="ani-graph-svg" viewBox="0 0 {width} {svg_height}" role="img">
        <g class="viewport">
          {''.join(edge_markup)}
          {''.join(node_markup)}
        </g>
      </svg>
    </div>
    <script>
      (function() {{
        const svg = document.getElementById("{graph_id}");
        const viewport = svg.querySelector(".viewport");
        let scale = 1;
        let tx = 0;
        let ty = 0;
        let dragging = false;
        let last = null;
        function apply() {{
          viewport.setAttribute("transform", `translate(${{tx}}, ${{ty}}) scale(${{scale}})`);
        }}
        svg.addEventListener("wheel", function(event) {{
          event.preventDefault();
          const delta = event.deltaY > 0 ? 0.9 : 1.1;
          scale = Math.min(3.2, Math.max(0.45, scale * delta));
          apply();
        }}, {{ passive: false }});
        svg.addEventListener("mousedown", function(event) {{
          dragging = true;
          last = [event.clientX, event.clientY];
          svg.style.cursor = "grabbing";
        }});
        window.addEventListener("mouseup", function() {{
          dragging = false;
          svg.style.cursor = "default";
        }});
        window.addEventListener("mousemove", function(event) {{
          if (!dragging || !last) return;
          tx += event.clientX - last[0];
          ty += event.clientY - last[1];
          last = [event.clientX, event.clientY];
          apply();
        }});
      }})();
    </script>
    """
    components.html(html_doc, height=height, scrolling=False)


def escape(value: object) -> str:
    """HTML-escape a value for safe component markup."""
    return html.escape(str(value), quote=True)


def _normalize_positions(
    positions: dict[str, tuple[float, float]],
    width: int,
    height: int,
) -> dict[str, tuple[float, float]]:
    """Scale spring-layout coordinates into SVG coordinates."""
    if not positions:
        return {}
    xs = [float(point[0]) for point in positions.values()]
    ys = [float(point[1]) for point in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 0.001)
    span_y = max(max_y - min_y, 0.001)
    padding_x = 78
    padding_y = 62
    normalized = {}
    for node_id, point in positions.items():
        x = padding_x + ((float(point[0]) - min_x) / span_x) * (width - padding_x * 2)
        y = padding_y + ((float(point[1]) - min_y) / span_y) * (height - padding_y * 2)
        normalized[node_id] = (x, y)
    return normalized


def _short_label(label: str, limit: int = 20) -> str:
    """Keep graph labels compact enough for dense network views."""
    if len(label) <= limit:
        return label
    return label[: limit - 1] + "."
