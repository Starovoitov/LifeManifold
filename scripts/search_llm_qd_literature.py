#!/usr/bin/env python3
"""Run the frozen literature searches and preserve raw metadata.

The script uses public metadata APIs only. It writes immutable, timestamped raw
exports plus normalized candidate and search-run JSONL files. Screening remains
a separate, explicit step. Search outputs stay under
``artifacts/controlled_attribution/`` as local design notes, not git artifacts.
"""

from __future__ import annotations

import argparse
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("artifacts/controlled_attribution")
CUTOFF_FILTER = {"from": "2022-01-01", "to": "2026-08-31"}
QUERIES = {
    "Q1": (
        '"large language model" ("quality diversity" OR "quality-diversity" '
        'OR "MAP-Elites") (evolution OR emitter OR mutation OR generator OR archive)'
    ),
    "Q2": (
        '"large language model" ("evolutionary search" OR '
        '"evolutionary computation" OR "genetic programming" OR "island model") '
        "(archive OR diversity OR baseline OR ablation OR allocation OR scheduler)"
    ),
    "Q3": (
        '("large language model" OR LLM) '
        '(evolution OR "quality diversity" OR "MAP-Elites") '
        '("adaptive operator" OR allocation OR routing OR scheduler OR UCB)'
    ),
}
AMENDED_QUERIES = {
    "openalex": (
        ("Q1a", "Q1", '"large language model" "quality diversity"'),
        ("Q1b", "Q1", '"large language model" "map elites"'),
        ("Q2a", "Q2", '"large language model" evolutionary search archive'),
        ("Q2b", "Q2", '"large language model" "island model" archive'),
        ("Q3a", "Q3", '"large language model" "adaptive operator selection"'),
        ("Q3b", "Q3", '"large language model" evolution allocation scheduler'),
    ),
    "semantic_scholar": (
        ("Q1a", "Q1", "large language model quality diversity"),
        ("Q1b", "Q1", "large language model map elites"),
        ("Q2a", "Q2", "large language model evolutionary search archive"),
        ("Q2b", "Q2", "large language model island model archive"),
        ("Q3a", "Q3", "large language model adaptive operator selection"),
        ("Q3b", "Q3", "large language model evolution allocation scheduler"),
    ),
    "dblp": (
        ("Q1a", "Q1", "large language model quality diversity"),
        ("Q1b", "Q1", "large language model map elites"),
        ("Q2a", "Q2", "large language model evolutionary search archive"),
        ("Q2b", "Q2", "large language model island model archive"),
        ("Q3a", "Q3", "large language model adaptive operator selection"),
        ("Q3b", "Q3", "large language model evolution allocation scheduler"),
    ),
}
ARXIV_QUERIES = {
    "Q1": (
        '(all:"large language model" OR all:LLM) AND '
        '(all:"quality diversity" OR all:"quality-diversity" OR all:"MAP-Elites")'
    ),
    "Q2": (
        '(all:"large language model" OR all:LLM) AND '
        '(all:"evolutionary search" OR all:"evolutionary computation" '
        'OR all:"genetic programming" OR all:"island model") AND '
        "(all:archive OR all:diversity OR all:ablation OR all:allocation)"
    ),
    "Q3": (
        '(all:"large language model" OR all:LLM) AND '
        '(all:evolution OR all:"quality diversity" OR all:"MAP-Elites") AND '
        '(all:"adaptive operator" OR all:allocation OR all:routing '
        "OR all:scheduler OR all:UCB)"
    ),
}


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "LifeManifold-ClaimInventory/1.0 (literature metadata search)",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _request_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "LifeManifold-ClaimInventory/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def search_openalex(query: str) -> tuple[list[dict[str, Any]], int, str]:
    base_filter = (
        f"from_publication_date:{CUTOFF_FILTER['from']},"
        f"to_publication_date:{CUTOFF_FILTER['to']},"
        f"title_and_abstract.search:{query}"
    )
    rows: list[dict[str, Any]] = []
    cursor = "*"
    total = 0
    first_url = ""
    while cursor:
        params = {"filter": base_filter, "per-page": "200", "cursor": cursor}
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        if not first_url:
            first_url = url
        payload = _request_json(url)
        total = int(payload.get("meta", {}).get("count", 0))
        rows.extend(payload.get("results", []))
        cursor = payload.get("meta", {}).get("next_cursor")
        if not payload.get("results") or len(rows) >= total:
            break
    return rows, total, first_url


def search_semantic_scholar(
    query: str,
) -> tuple[list[dict[str, Any]], int, str]:
    params = {
        "query": query,
        "year": "2022-2026",
        "limit": "100",
        "fields": "title,authors,year,externalIds,url,abstract,venue",
    }
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search/bulk?"
        + urllib.parse.urlencode(params)
    )
    payload = _request_json(url)
    return payload.get("data", []), int(payload.get("total", 0)), url


def search_dblp(query: str) -> tuple[list[dict[str, Any]], int, str]:
    params = {"q": query, "h": "1000", "format": "json"}
    url = "https://dblp.org/search/publ/api?" + urllib.parse.urlencode(params)
    payload = _request_json(url)
    result = payload.get("result", {})
    hits = result.get("hits", {}).get("hit", [])
    return hits, int(result.get("hits", {}).get("@total", 0)), url


def search_arxiv(query: str) -> tuple[list[dict[str, Any]], int, str]:
    # Atom XML is preserved verbatim. Candidate normalization is intentionally
    # conservative and extracts only entry-level identity fields.
    params = {
        "search_query": query,
        "start": "0",
        "max_results": "200",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    xml = _request_text(url)
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    atom = "{http://www.w3.org/2005/Atom}"
    opensearch = "{http://a9.com/-/spec/opensearch/1.1/}"
    total_text = root.findtext(f"{opensearch}totalResults", default="0")
    rows: list[dict[str, Any]] = []
    for entry in root.findall(f"{atom}entry"):
        rows.append(
            {
                "id": entry.findtext(f"{atom}id", default=""),
                "title": " ".join(entry.findtext(f"{atom}title", default="").split()),
                "summary": " ".join(
                    entry.findtext(f"{atom}summary", default="").split()
                ),
                "published": entry.findtext(f"{atom}published", default=""),
                "updated": entry.findtext(f"{atom}updated", default=""),
                "authors": [
                    author.findtext(f"{atom}name", default="")
                    for author in entry.findall(f"{atom}author")
                ],
            }
        )
    return rows, int(total_text), url


def _candidate_from_row(
    source: str, query_family: str, row: dict[str, Any]
) -> dict[str, Any] | None:
    if source == "openalex":
        title = row.get("title")
        year = row.get("publication_year")
        identifiers = row.get("ids", {})
        url = row.get("id")
        abstract = ""
    elif source == "semantic_scholar":
        title = row.get("title")
        year = row.get("year")
        identifiers = row.get("externalIds", {})
        url = row.get("url")
        abstract = row.get("abstract") or ""
    elif source == "dblp":
        info = row.get("info", {})
        title = info.get("title")
        try:
            year = int(info.get("year")) if info.get("year") else None
        except ValueError:
            year = None
        identifiers = {"dblp_key": info.get("key"), "doi": info.get("doi")}
        url = info.get("url")
        abstract = ""
    else:
        title = row.get("title")
        published = str(row.get("published", ""))
        year = int(published[:4]) if published[:4].isdigit() else None
        identifiers = {"arxiv": str(row.get("id", "")).rsplit("/", 1)[-1]}
        url = row.get("id")
        abstract = row.get("summary", "")
    if not title:
        return None
    return {
        "record_type": "candidate",
        "title": " ".join(str(title).split()),
        "year": year,
        "source": source,
        "query_family": query_family,
        "identifiers": identifiers,
        "url": url,
        "abstract": abstract,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("openalex", "semantic_scholar", "dblp", "arxiv"),
        default=("openalex", "semantic_scholar", "dblp", "arxiv"),
    )
    parser.add_argument(
        "--amendment-1",
        action="store_true",
        help="Use the focused source-specific queries frozen in Amendment 1.",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    raw_dir = args.root / "raw" / stamp
    raw_dir.mkdir(parents=True, exist_ok=False)
    search_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    functions = {
        "openalex": search_openalex,
        "semantic_scholar": search_semantic_scholar,
        "dblp": search_dblp,
        "arxiv": search_arxiv,
    }

    for source in args.sources:
        if args.amendment_1 and source != "arxiv":
            source_queries = AMENDED_QUERIES[source]
        else:
            source_queries = tuple(
                (
                    family,
                    family,
                    ARXIV_QUERIES[family] if source == "arxiv" else default_query,
                )
                for family, default_query in QUERIES.items()
            )
        for query_name, family, query in source_queries:
            search_id = f"{stamp.lower()}-{source}-{query_name.lower()}"
            raw_path = raw_dir / f"{source}_{query_name.lower()}.json"
            try:
                rows, total, url = functions[source](query)
                status = "ok"
                error = None
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                ValueError,
            ) as exc:
                rows, total, url = [], 0, ""
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
            raw_path.write_text(
                json.dumps(
                    {
                        "search_id": search_id,
                        "status": status,
                        "error": error,
                        "source": source,
                        "query_family": family,
                        "query": query,
                        "request_url": url,
                        "reported_total": total,
                        "returned_count": len(rows),
                        "rows": rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            complete = status == "ok" and len(rows) >= total
            search_rows.append(
                {
                    "record_type": "search_run",
                    "search_id": search_id,
                    "executed_at": timestamp.isoformat().replace("+00:00", "Z"),
                    "source": source,
                    "query_family": family,
                    "exact_query": query,
                    "filters": CUTOFF_FILTER,
                    "result_count": total,
                    "formal": complete,
                    "export_path": str(raw_path),
                    "notes": (
                        f"returned={len(rows)}; amendment=1"
                        if complete
                        else (
                            f"incomplete: returned={len(rows)} of reported={total}"
                            if status == "ok"
                            else f"search failed and is not formal: {error}"
                        )
                    ),
                }
            )
            for row in rows:
                candidate = _candidate_from_row(source, family, row)
                if candidate is not None:
                    candidate_rows.append(candidate)
            time.sleep(1)

    with (args.root / "search_runs.jsonl").open("a", encoding="utf-8") as handle:
        for row in search_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    candidate_path = raw_dir / "candidates.jsonl"
    candidate_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in candidate_rows
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(search_rows)} search runs and "
        f"{len(candidate_rows)} source records to {raw_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
