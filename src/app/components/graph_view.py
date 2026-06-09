"""Reusable graph evidence components for the Streamlit product demo."""

from __future__ import annotations

import hashlib
import html

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


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

EDGE_LABELS = {
    "represents": "Represented by",
    "included_in": "Included in exhibition",
    "hosted_by": "Hosted by",
    "gallery_exhibition": "Gallery exhibition",
    "acquired_artist": "Acquired by",
    "acquired_work_by": "Acquired by",
    "collects": "Collected by",
    "curated_artist": "Curated by",
    "curated": "Curated by",
    "has_auction_result": "Auction result",
    "mentioned_in_press": "Press mention",
    "made_acquisition": "Acquisition",
}

NODE_TYPE_LABELS = {
    "artist": "Artist",
    "gallery": "Gallery",
    "museum": "Museum",
    "collector": "Collector",
    "curator": "Curator",
    "exhibition": "Exhibition",
    "acquisition": "Acquisition",
    "auction_result": "Auction result",
    "press_mentions": "Press mention",
    "institution": "Institution",
    "artwork": "Artwork",
    "genre": "Genre",
}

SIGNAL_WEIGHTS = {
    "acquired_artist": 10,
    "acquired_work_by": 8,
    "included_in": 7,
    "hosted_by": 5,
    "represents": 6,
    "collects": 5,
    "curated_artist": 5,
    "curated": 4,
}

SIGNAL_TYPES = set(SIGNAL_WEIGHTS)


def get_artist_ego_network(
    graph: nx.MultiDiGraph,
    artist_id: str,
    max_nodes: int = 24,
    comparable_artist_ids: list[str] | None = None,
) -> nx.MultiDiGraph:
    """Return a compact, signal-ranked ego network for an artist."""
    if artist_id not in graph:
        return nx.MultiDiGraph()

    comparable_artist_ids = comparable_artist_ids or []
    selected_edges = get_top_signal_edges(graph, artist_id, limit=max_nodes)
    selected_nodes = {artist_id}
    for source, target, _ in selected_edges:
        selected_nodes.update((source, target))
        selected_nodes.update(_supporting_context_nodes(graph, source, target))

    ranked_nodes = _rank_nodes_for_artist(graph, artist_id, selected_nodes)
    limited_nodes = set(ranked_nodes[:max_nodes])
    limited_nodes.add(artist_id)
    for comparable_id in comparable_artist_ids[:5]:
        if comparable_id in graph and nx.has_path(nx.Graph(graph), artist_id, comparable_id) and len(limited_nodes) < max_nodes:
            limited_nodes.add(comparable_id)

    return graph.subgraph(limited_nodes).copy()


def get_top_signal_edges(
    graph: nx.MultiDiGraph,
    artist_id: str,
    limit: int = 10,
) -> list[tuple[str, str, dict[str, object]]]:
    """Return the strongest score-relevant edges touching an artist."""
    if artist_id not in graph:
        return []
    edges = []
    for source, target, data in graph.in_edges(artist_id, data=True):
        if data.get("relationship_type") in SIGNAL_TYPES:
            edges.append((source, target, data))
    for source, target, data in graph.out_edges(artist_id, data=True):
        if data.get("relationship_type") in SIGNAL_TYPES:
            edges.append((source, target, data))
    return sorted(edges, key=_edge_signal_score, reverse=True)[:limit]


def filter_graph_by_date(graph: nx.MultiDiGraph, cutoff_date: str) -> nx.MultiDiGraph:
    """Return graph edges and event nodes available on or before a cutoff date."""
    cutoff = pd.Timestamp(cutoff_date)
    filtered = nx.MultiDiGraph()
    filtered.add_nodes_from(graph.nodes(data=True))
    for source, target, data in graph.edges(data=True):
        if pd.Timestamp(data.get("start_date")) <= cutoff:
            filtered.add_edge(source, target, **data)

    connected_nodes = {source for source, target in filtered.edges()} | {target for source, target in filtered.edges()}
    entity_nodes = {
        node_id
        for node_id, node_data in graph.nodes(data=True)
        if node_data.get("node_type") in {"artist", "gallery", "museum", "collector", "curator"}
    }
    return filtered.subgraph(connected_nodes | entity_nodes).copy()


def filter_graph_by_confidence(graph: nx.MultiDiGraph, min_confidence: float) -> nx.MultiDiGraph:
    """Return graph edges with confidence above a minimum threshold."""
    filtered = nx.MultiDiGraph()
    filtered.add_nodes_from(graph.nodes(data=True))
    for source, target, data in graph.edges(data=True):
        if float(data.get("confidence_score", 0) or 0) >= min_confidence:
            filtered.add_edge(source, target, **data)
    connected_nodes = {source for source, target in filtered.edges()} | {target for source, target in filtered.edges()}
    return filtered.subgraph(connected_nodes).copy()


def render_graph_view(
    graph: nx.MultiDiGraph,
    selected_node: str | None = None,
    highlighted_nodes: dict[str, set[str]] | None = None,
    max_edges: int = 80,
    height: int = 560,
    show_edge_labels: bool = True,
) -> None:
    """Render a compact SVG graph with labels, tooltips, pan, and zoom."""
    if graph.number_of_nodes() == 0:
        st.info("No graph data matches the current filters.")
        return

    highlighted_nodes = highlighted_nodes or {}
    visible_edges = list(graph.edges(data=True))[:max_edges]
    visible_nodes = set(graph.nodes())
    if visible_edges:
        visible_nodes = {source for source, _, _ in visible_edges} | {target for _, target, _ in visible_edges}
        visible_nodes.update(node for nodes in highlighted_nodes.values() for node in nodes if node in graph)

    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(visible_nodes)
    layout_graph.add_edges_from((source, target) for source, target, _ in visible_edges)
    positions = nx.spring_layout(layout_graph, seed=11, k=1.25)
    width = 980
    svg_height = max(360, height - 82)
    normalized = _normalize_positions(positions, width, svg_height)
    graph_id = "ani_graph_" + hashlib.md5("|".join(sorted(visible_nodes)).encode("utf-8")).hexdigest()[:10]

    edge_markup = []
    label_markup = []
    for source, target, edge_data in visible_edges:
        if source not in normalized or target not in normalized:
            continue
        x1, y1 = normalized[source]
        x2, y2 = normalized[target]
        label = _edge_display_label(str(edge_data.get("relationship_type", "")))
        confidence = float(edge_data.get("confidence_score", 0.0) or 0.0)
        edge_markup.append(
            f"""
            <line class="ani-graph-edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"
              stroke-width="{1.0 + confidence:.2f}">
              <title>{_escape(_node_label(graph, source))} -> {_escape(_node_label(graph, target))} | {_escape(label)} | confidence {confidence:.2f}</title>
            </line>
            """
        )
        if show_edge_labels and label and len(visible_edges) <= 35:
            label_markup.append(
                f'<text class="ani-edge-label" x="{((x1 + x2) / 2):.1f}" y="{((y1 + y2) / 2):.1f}" text-anchor="middle">{_escape(label)}</text>'
            )

    node_markup = []
    for node_id in visible_nodes:
        if node_id not in normalized:
            continue
        x, y = normalized[node_id]
        data = graph.nodes[node_id]
        node_type = str(data.get("node_type", "unknown"))
        name = _node_label(graph, node_id)
        color = NODE_COLORS.get(node_type, "#8f8a80")
        role = _highlight_role(node_id, highlighted_nodes)
        radius = 16 if node_id == selected_node else 12
        stroke = _highlight_stroke(role, node_id == selected_node)
        stroke_width = 3 if role or node_id == selected_node else 1.5
        node_markup.append(
            f"""
            <g class="ani-graph-node">
              <circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}">
                <title>{_escape(name)} | {_escape(node_type_display_label(node_type))} | {_escape(node_id)}</title>
              </circle>
              <text x="{x:.1f}" y="{(y + radius + 15):.1f}" text-anchor="middle">{_escape(_short_label(name))}</text>
            </g>
            """
        )

    html_doc = f"""
    {_graph_styles()}
    <div class="ani-graph-shell">
      <div class="ani-graph-toolbar">
        <span>Evidence network. Scroll to zoom. Drag to pan.</span>
        <span>{len(visible_nodes)} nodes / {len(visible_edges)} relationships</span>
      </div>
      <svg id="{graph_id}" class="ani-graph-svg" viewBox="0 0 {width} {svg_height}" role="img">
        <g class="viewport">
          {''.join(edge_markup)}
          {''.join(label_markup)}
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
          scale = Math.min(3.1, Math.max(0.55, scale * (event.deltaY > 0 ? 0.9 : 1.1)));
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


def graph_signal_summary(graph: nx.MultiDiGraph, artist_id: str) -> pd.DataFrame:
    """Summarize graph-derived evidence for the selected artist."""
    top_edges = get_top_signal_edges(graph, artist_id, limit=10)
    strongest = top_edges[0] if top_edges else None
    closest_major = _closest_node_by_type(graph, artist_id, "museum")
    closest_collector = _closest_node_by_type(graph, artist_id, "collector")
    rows = [
        {
            "signal": "strongest relationship",
            "detail": _relationship_summary(graph, strongest) if strongest else "No relationship evidence available",
        },
        {
            "signal": "closest major institution",
            "detail": closest_major,
        },
        {
            "signal": "influential collector proximity",
            "detail": closest_collector,
        },
        {
            "signal": "graph-derived score drivers",
            "detail": ", ".join(sorted({_edge_display_label(str(edge[2].get("relationship_type", ""))) for edge in top_edges if _edge_display_label(str(edge[2].get("relationship_type", "")))})),
        },
    ]
    return pd.DataFrame(rows)


def highlighted_evidence_nodes(
    graph: nx.MultiDiGraph,
    artist_id: str,
    comparable_artist_ids: list[str] | None = None,
) -> dict[str, set[str]]:
    """Return node sets to highlight in the graph renderer."""
    comparable_artist_ids = comparable_artist_ids or []
    highlights = {
        "current_gallery": set(),
        "museums": set(),
        "collectors": set(),
        "curators": set(),
        "comparables": {node_id for node_id in comparable_artist_ids if node_id in graph},
    }
    for source, target, data in graph.in_edges(artist_id, data=True):
        node_type = graph.nodes[source].get("node_type")
        relationship = data.get("relationship_type")
        if relationship == "represents":
            highlights["current_gallery"].add(source)
        elif node_type == "museum":
            highlights["museums"].add(source)
        elif node_type == "collector":
            highlights["collectors"].add(source)
        elif node_type == "curator":
            highlights["curators"].add(source)
    for source, target, data in graph.out_edges(artist_id, data=True):
        target_type = graph.nodes[target].get("node_type")
        if target_type == "exhibition":
            for _, museum_id, edge_data in graph.out_edges(target, data=True):
                if edge_data.get("relationship_type") == "hosted_by":
                    highlights["museums"].add(museum_id)
        elif target_type == "museum":
            highlights["museums"].add(target)
    return highlights


def _rank_nodes_for_artist(graph: nx.MultiDiGraph, artist_id: str, nodes: set[str]) -> list[str]:
    """Rank ego-network nodes by evidence relevance."""
    scores = {node_id: 0.0 for node_id in nodes}
    scores[artist_id] = 999
    for source, target, data in graph.edges(data=True):
        if source not in nodes or target not in nodes:
            continue
        score = _edge_signal_score((source, target, data))
        scores[source] = scores.get(source, 0) + score
        scores[target] = scores.get(target, 0) + score
    return sorted(nodes, key=lambda node_id: scores.get(node_id, 0), reverse=True)


def _supporting_context_nodes(graph: nx.MultiDiGraph, source: str, target: str) -> set[str]:
    """Include only context nodes that explain an artist signal."""
    context = set()
    for node_id in (source, target):
        if node_id not in graph:
            continue
        node_type = graph.nodes[node_id].get("node_type")
        if node_type == "exhibition":
            for _, museum_id, data in graph.out_edges(node_id, data=True):
                if data.get("relationship_type") == "hosted_by":
                    context.add(museum_id)
        elif node_type == "acquisition":
            for museum_id, _, data in graph.in_edges(node_id, data=True):
                if data.get("relationship_type") == "made_acquisition":
                    context.add(museum_id)
    return context


def _edge_signal_score(edge: tuple[str, str, dict[str, object]]) -> float:
    """Score edge relevance for artist diligence."""
    _, _, data = edge
    relationship = str(data.get("relationship_type", ""))
    confidence = float(data.get("confidence_score", 0.0) or 0.0)
    return SIGNAL_WEIGHTS.get(relationship, 1) + confidence


def _edge_display_label(relationship_type: str) -> str:
    """Map internal relationship names to collector-facing labels."""
    return relationship_display_label(relationship_type)


def relationship_display_label(relationship_type: str) -> str:
    """Map internal relationship names to collector-facing labels."""
    return EDGE_LABELS.get(relationship_type, humanize_identifier(relationship_type))


def node_type_display_label(node_type: str) -> str:
    """Map internal node types to collector-facing labels."""
    return NODE_TYPE_LABELS.get(node_type, humanize_identifier(node_type))


def humanize_identifier(value: object) -> str:
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

    prefix = parts[0].lower()
    entity_prefixes = {"artist", "gallery", "museum", "collector", "curator", "institution"}
    event_prefixes = {
        "auction": "Auction result",
        "press": "Press mention",
        "event": "Event",
        "fair": "Art fair",
        "acq": "Acquisition",
        "rel": "Relationship",
    }

    if prefix in entity_prefixes:
        return _title_identifier_text(" ".join(parts[1:]))
    if prefix in event_prefixes:
        detail_parts = parts[1:]
        detail_parts = [part for part in detail_parts if part.lower() not in entity_prefixes]
        detail = _title_identifier_text(" ".join(detail_parts))
        return f"{event_prefixes[prefix]} - {detail}" if detail else event_prefixes[prefix]
    return _title_identifier_text(" ".join(parts))


def _title_identifier_text(text: str) -> str:
    """Title-case identifier text while preserving common abbreviations."""
    abbreviations = {"id": "ID", "url": "URL", "usd": "USD", "1y": "1Y", "3y": "3Y"}
    return " ".join(abbreviations.get(part.lower(), part.capitalize()) for part in text.split())


def _closest_node_by_type(graph: nx.MultiDiGraph, artist_id: str, node_type: str) -> str:
    """Return closest node of a given type in the undirected graph."""
    if artist_id not in graph:
        return "Not available"
    undirected = nx.Graph(graph)
    candidates = [node_id for node_id, data in graph.nodes(data=True) if data.get("node_type") == node_type]
    distances = []
    for node_id in candidates:
        try:
            distances.append((nx.shortest_path_length(undirected, artist_id, node_id), _node_label(graph, node_id)))
        except nx.NetworkXNoPath:
            continue
    if not distances:
        return "No connected node found"
    distance, label = sorted(distances)[0]
    return f"{label} at graph distance {distance}"


def _relationship_summary(graph: nx.MultiDiGraph, edge: tuple[str, str, dict[str, object]]) -> str:
    """Return a readable summary for one edge."""
    source, target, data = edge
    return f"{_node_label(graph, source)} -> {_node_label(graph, target)} ({_edge_display_label(str(data.get('relationship_type', 'relationship')))})"


def _node_label(graph: nx.MultiDiGraph, node_id: str) -> str:
    """Return display label for one graph node."""
    data = graph.nodes[node_id]
    return str(data.get("name") or data.get("title") or humanize_identifier(node_id))


def _highlight_role(node_id: str, highlighted_nodes: dict[str, set[str]]) -> str:
    """Return first highlight role for a node."""
    for role, nodes in highlighted_nodes.items():
        if node_id in nodes:
            return role
    return ""


def _highlight_stroke(role: str, selected: bool) -> str:
    """Return node stroke color for highlight roles."""
    if selected:
        return "#181716"
    return {
        "current_gallery": "#181716",
        "museums": "#3f654f",
        "collectors": "#6d5a78",
        "curators": "#8a6d32",
        "comparables": "#884339",
    }.get(role, "#ffffff")


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
    padding_x = 80
    padding_y = 64
    return {
        node_id: (
            padding_x + ((float(point[0]) - min_x) / span_x) * (width - padding_x * 2),
            padding_y + ((float(point[1]) - min_y) / span_y) * (height - padding_y * 2),
        )
        for node_id, point in positions.items()
    }


def _short_label(label: str, limit: int = 20) -> str:
    """Keep graph labels compact enough for dense network views."""
    if len(label) <= limit:
        return label
    return label[: limit - 1] + "."


def _escape(value: object) -> str:
    """HTML-escape graph text."""
    return html.escape(str(value), quote=True)


def _graph_styles() -> str:
    """Return iframe-local graph styling."""
    return """
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
        height: 478px;
        width: 100%;
      }
      .ani-graph-edge {
        stroke: var(--ani-graph-border-strong);
        stroke-opacity: .72;
      }
      .ani-edge-label {
        fill: var(--ani-graph-muted);
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 10px;
        paint-order: stroke;
        pointer-events: none;
        stroke: var(--ani-graph-surface-alt);
        stroke-width: 4px;
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
