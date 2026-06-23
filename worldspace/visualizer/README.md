# Deprecated: matplotlib PNG visualizer

**Status:** deprecated as of dashboard MVP. Do not add new features here.

## Use instead

```bash
cd dashboard
streamlit run Home.py
```

- MAP-Elites archives: **Archive Explorer**, **Metrics Dashboard**, **Surrogate Analysis**
- Training buffer: **Training Buffer** page
- LLM prompts: **LLM Prompt Tester**

Dashboard setup: [docs/DASHBOARD.md](../../docs/DASHBOARD.md).

## Still supported (legacy)

Headless PNG export for **pipeline** metrics JSONL and CA step traces:

```bash
uv run python -m worldspace.visualizer \
  --output-dir results/plots \
  --metrics-jsonl results/trace.jsonl
```

Imports: `worldspace.visualizer.plotting` and `worldspace.visualizer.diagnostics` (not re-exported from top-level `worldspace`).
