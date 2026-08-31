#!/usr/bin/env python3
"""Deduplicate formal literature exports and prepare auditable screening rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("artifacts/controlled_attribution")
LLM_TERMS = ("large language model", "llm")
QD_TERMS = (
    "quality diversity",
    "quality-diversity",
    "map-elites",
    "map elites",
    "illumination algorithm",
    "repertoire",
)
ADJACENT_TERMS = (
    "evolutionary search",
    "evolutionary computation",
    "genetic programming",
    "island model",
    "evolutionary inference",
    "adaptive model selection",
    "adaptive operator",
)
EMPIRICAL_TERMS = (
    "evaluate",
    "evaluation",
    "experiment",
    "benchmark",
    "outperform",
    "compared",
    "comparison",
    "ablation",
    "results",
    "versus",
    " vs ",
)
SECONDARY_TITLE_TERMS = (
    "survey",
    "review",
    "roadmap",
    "perspective",
    "bibliometric",
)


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def slug(value: str, limit: int = 64) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized[:limit].rstrip("-") or "untitled"


def normalize_doi(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).casefold().strip()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    return text or None


def reconstruct_openalex_abstract(row: dict[str, Any]) -> str:
    inverted = row.get("abstract_inverted_index")
    if not isinstance(inverted, dict):
        return ""
    positioned: list[tuple[int, str]] = []
    for token, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        positioned.extend((int(position), str(token)) for position in positions)
    return " ".join(token for _, token in sorted(positioned))


def normalize_raw_row(source: str, row: dict[str, Any]) -> dict[str, Any] | None:
    if source == "openalex":
        title = row.get("title")
        year = row.get("publication_year")
        ids = row.get("ids") or {}
        doi = normalize_doi(ids.get("doi") or row.get("doi"))
        arxiv = None
        abstract = reconstruct_openalex_abstract(row)
        authors = [
            str((authorship.get("author") or {}).get("display_name"))
            for authorship in row.get("authorships", [])
            if (authorship.get("author") or {}).get("display_name")
        ]
        venue = str(
            ((row.get("primary_location") or {}).get("source") or {}).get(
                "display_name"
            )
            or ""
        )
        url = (row.get("primary_location") or {}).get("landing_page_url") or row.get(
            "id"
        )
    elif source == "semantic_scholar":
        title = row.get("title")
        year = row.get("year")
        ids = row.get("externalIds") or {}
        doi = normalize_doi(ids.get("DOI"))
        arxiv = ids.get("ArXiv")
        abstract = row.get("abstract") or ""
        authors = [
            str(author.get("name"))
            for author in row.get("authors", [])
            if author.get("name")
        ]
        venue = str(row.get("venue") or "")
        url = row.get("url")
    elif source == "dblp":
        info = row.get("info") or {}
        title = info.get("title")
        year = info.get("year")
        doi = normalize_doi(info.get("doi"))
        arxiv = None
        abstract = ""
        raw_authors = info.get("authors", {}).get("author", [])
        if isinstance(raw_authors, dict):
            raw_authors = [raw_authors]
        authors = [
            str(author.get("text") if isinstance(author, dict) else author)
            for author in raw_authors
        ]
        venue = str(info.get("venue") or "")
        url = info.get("url")
    elif source == "arxiv":
        title = row.get("title")
        published = str(row.get("published") or "")
        year = int(published[:4]) if published[:4].isdigit() else None
        doi = None
        arxiv = str(row.get("id") or "").rsplit("/", 1)[-1].split("v", 1)[0]
        abstract = row.get("summary") or ""
        authors = [str(author) for author in row.get("authors", []) if author]
        venue = "arXiv"
        url = row.get("id")
    else:
        return None
    if not title:
        return None
    return {
        "title": " ".join(str(title).split()),
        "normalized_title": normalize_title(str(title)),
        "year": int(year) if str(year).isdigit() else None,
        "doi": doi,
        "arxiv": arxiv,
        "abstract": " ".join(str(abstract).split()),
        "authors": authors,
        "venue": venue,
        "url": url,
        "sources": [source],
        "query_families": [],
    }


def classify(candidate: dict[str, Any]) -> tuple[str, str | None, str]:
    title = candidate["title"].casefold()
    text = f"{title} {candidate.get('abstract', '').casefold()}"
    if any(term in title for term in SECONDARY_TITLE_TERMS):
        return "exclude", "secondary", "Title identifies a review/survey."
    if not any(term in text for term in LLM_TERMS):
        return "exclude", "no_llm_component", "No LLM component in title/abstract."
    has_qd = any(term in text for term in QD_TERMS)
    has_adjacent = any(term in text for term in ADJACENT_TERMS)
    if not has_qd and not has_adjacent:
        return (
            "exclude",
            "no_qd_or_adjacent",
            "No explicit QD/MAP-Elites or eligible adjacent evolutionary context.",
        )
    has_empirical = any(term in text for term in EMPIRICAL_TERMS)
    if not has_empirical and not candidate.get("abstract"):
        return (
            "unclear",
            None,
            "Relevant title but metadata export lacks an abstract.",
        )
    if not has_empirical:
        return (
            "unclear",
            None,
            "Relevant method; empirical focal-vs-baseline contrast needs full text.",
        )
    return (
        "include",
        None,
        "Title/abstract indicates eligible context and an empirical comparison.",
    )


def load_formal_exports(root: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    formal_runs: list[dict[str, Any]] = []
    for line in (root / "search_runs.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        if run.get("formal"):
            formal_runs.append(run)
    exports: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for run in formal_runs:
        payload = json.loads(Path(run["export_path"]).read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            exports.append((run, row))
    return exports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    deduplicated: list[dict[str, Any]] = []
    key_to_index: dict[str, int] = {}
    duplicate_rows: list[dict[str, Any]] = []
    for run, raw in load_formal_exports(args.root):
        candidate = normalize_raw_row(run["source"], raw)
        if candidate is None:
            continue
        candidate["query_families"] = [run["query_family"]]
        keys = [
            f"doi:{candidate['doi']}" if candidate["doi"] else None,
            f"arxiv:{candidate['arxiv']}" if candidate["arxiv"] else None,
            f"title:{candidate['normalized_title']}:{candidate['year']}",
        ]
        existing = next(
            (key_to_index[key] for key in keys if key in key_to_index), None
        )
        if existing is not None:
            retained = deduplicated[existing]
            retained["sources"] = sorted(
                set(retained["sources"]) | set(candidate["sources"])
            )
            retained["query_families"] = sorted(
                set(retained["query_families"]) | set(candidate["query_families"])
            )
            if not retained.get("abstract") and candidate.get("abstract"):
                retained["abstract"] = candidate["abstract"]
            if not retained.get("doi") and candidate.get("doi"):
                retained["doi"] = candidate["doi"]
            if not retained.get("arxiv") and candidate.get("arxiv"):
                retained["arxiv"] = candidate["arxiv"]
            if not retained.get("authors") and candidate.get("authors"):
                retained["authors"] = candidate["authors"]
            if not retained.get("venue") and candidate.get("venue"):
                retained["venue"] = candidate["venue"]
            duplicate_rows.append(
                {
                    "record_type": "screening",
                    "screening_id": (
                        f"dup-{len(duplicate_rows) + 1:04d}-"
                        f"{slug(candidate['title'], 40)}"
                    ),
                    "title": candidate["title"],
                    "discovery_source": "+".join(candidate["sources"]),
                    "stage": "title_abstract",
                    "decision": "exclude",
                    "reason_code": "duplicate",
                    "audited": True,
                    "notes": f"Deduplicated into {retained['title']}.",
                }
            )
            continue
        index = len(deduplicated)
        for key in keys:
            if key:
                key_to_index[key] = index
        deduplicated.append(candidate)

    screening_rows = duplicate_rows[:]
    for index, candidate in enumerate(deduplicated, 1):
        decision, reason, notes = classify(candidate)
        screening_id = f"ta-{index:04d}-{slug(candidate['title'], 48)}"
        candidate["candidate_id"] = f"candidate-{index:04d}"
        candidate["screening_id"] = screening_id
        candidate["initial_decision"] = decision
        candidate["initial_reason_code"] = reason
        screening_rows.append(
            {
                "record_type": "screening",
                "screening_id": screening_id,
                "title": candidate["title"],
                "discovery_source": "+".join(candidate["sources"]),
                "stage": "title_abstract",
                "decision": decision,
                "reason_code": reason,
                "audited": False,
                "notes": notes,
            }
        )

    candidates_path = args.root / "screening_candidates.jsonl"
    candidates_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in deduplicated
        ),
        encoding="utf-8",
    )
    screening_path = args.root / "screening.jsonl"
    screening_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in screening_rows
        ),
        encoding="utf-8",
    )
    report_lines = [
        "# Title/abstract screening candidates",
        "",
        "Machine-assisted high-recall pass. `include` and `unclear` require "
        "full-text adjudication; they are not final inclusions.",
        "",
    ]
    for decision in ("include", "unclear"):
        report_lines.extend([f"## {decision}", ""])
        for candidate in deduplicated:
            if candidate["initial_decision"] != decision:
                continue
            report_lines.append(
                f"- `{candidate['candidate_id']}` "
                f"{candidate['title']} ({candidate.get('year') or 'year unclear'}) "
                f"[{', '.join(candidate['sources'])}]"
            )
        report_lines.append("")
    (args.root / "TITLE_ABSTRACT_SCREENING.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for row in screening_rows:
        key = f"{row['decision']}:{row.get('reason_code')}"
        counts[key] = counts.get(key, 0) + 1
    print(
        f"Formal source records={len(deduplicated) + len(duplicate_rows)}; "
        f"duplicates={len(duplicate_rows)}; unique={len(deduplicated)}"
    )
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
