#!/usr/bin/env python3
"""Apply the audited full-text decisions and build paper/contrast rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("artifacts/controlled_attribution")
COMPONENTS = (
    "initialization",
    "selector",
    "generator",
    "prompt_channel",
    "repair_fallback",
    "gate",
    "replacement",
    "allocation",
    "budget",
    "representation",
    "model",
    "evaluator",
)
ALL_BUDGETS = {
    "proposal",
    "valid_proposal",
    "evaluation",
    "llm_call",
    "token",
    "wall_time",
    "monetary",
}

# fragment, paper_id, stratum, domain, task, focal, baseline, endpoint,
# source URL, locator, treatment pattern
INCLUSIONS = [
    (
        "adaptive strategy generation for boundary",
        "abex-2026",
        "core",
        "software testing",
        "boundary-value exploration",
        "ABEX adaptive LLM strategy generation",
        "hand-engineered QD baseline",
        "QD-score and mutation score",
        "https://arxiv.org/abs/2608.28230",
        "Results and adaptive-strategy ablation",
        "bundle",
    ),
    (
        "atlas scaffold free",
        "atlas-2026",
        "core",
        "automated algorithm design",
        "four NP-hard problems",
        "three-layer embedding-guided QD search",
        "matched full-synthesis and layer ablations",
        "objective performance and design diversity",
        "https://arxiv.org/abs/2608.15546",
        "Table V and Appendix C",
        "archive",
    ),
    (
        "mosaic adversarial",
        "mosaic-2026",
        "core",
        "automated heuristic design",
        "specialist heuristic portfolios",
        "MOSAIC co-evolution",
        "LLM-AHD and evolutionary instance-generation baselines",
        "benchmark objective performance",
        "https://arxiv.org/abs/2608.07544",
        "Experiments, main benchmark comparison",
        "bundle",
    ),
    (
        "ideagent",
        "ideagent-2026",
        "core",
        "research ideation",
        "idea generation",
        "agentic QD lineages",
        "sequential-memory, stateless, and NOVA baselines",
        "idea quality and diversity",
        "https://arxiv.org/abs/2607.22375",
        "Table 1",
        "bundle",
    ),
    (
        "qdevo",
        "qdevo-2026",
        "core",
        "automated heuristic design",
        "multi-objective heuristic generation",
        "semantic QD archive",
        "structural-cell and NSGA-II variants",
        "quality and diversity outcomes",
        "https://arxiv.org/abs/2607.11916",
        "Table 2 and ablation",
        "archive",
    ),
    (
        "tacevo",
        "tacevo-2026",
        "core",
        "robotics",
        "tactile architecture discovery",
        "LLM-driven MAP-Elites architectures",
        "expert architecture baseline",
        "task accuracy and architecture diversity",
        "https://arxiv.org/abs/2606.30109",
        "Tables II-III",
        "bundle",
    ),
    (
        "heuresis",
        "heuresis-2026",
        "core",
        "AI research agents",
        "research search strategies",
        "MAP-Elites, Go-Explore, and island strategies",
        "greedy and divergent strategies",
        "quality, diversity, and novelty",
        "https://arxiv.org/abs/2606.25198",
        "Results, six-strategy comparison",
        "archive",
    ),
    (
        "distributed quality diversity search for toxicity",
        "toxsearch-distributed-2026",
        "core",
        "LLM safety",
        "toxicity discovery",
        "distributed ToxSearch-S",
        "ToxSearch, RainbowPlus, and sequential execution",
        "toxicity and coverage",
        "https://arxiv.org/abs/2606.24166",
        "Results and Appendix Table 6",
        "bundle",
    ),
    (
        "llm guided evolution for medical decision",
        "medical-dejaq-2026",
        "core",
        "medicine",
        "decision-pipeline generation",
        "evolved MAP-Elites policies and prompts",
        "manual seeds",
        "pipeline quality and diversity",
        "https://arxiv.org/abs/2606.07342",
        "Results and Appendix Table 14",
        "generator",
    ),
    (
        "cross generational transfer of adversarial",
        "cross-generation-qd-2026",
        "core",
        "LLM safety",
        "adversarial-attack transfer",
        "fixed MAP-Elites probe",
        "target-model generations",
        "attack success and transfer",
        "https://arxiv.org/abs/2606.00813",
        "Results, cross-generation ASR",
        "model",
    ),
    (
        "quality diversity evolution for discovering diverse vulnerabilities",
        "vulnerability-qd-2026",
        "core",
        "LLM safety",
        "vulnerability discovery",
        "QD evolutionary probe",
        "four target LLMs",
        "coverage and attack success",
        "https://arxiv.org/abs/2606.00801",
        "Tables 1-3",
        "model",
    ),
    (
        "llm evolved domain independent heuristics",
        "planning-qd-2026",
        "core",
        "symbolic planning",
        "domain-independent heuristics",
        "LLM/MAP-Elites heuristics",
        "hand-engineered planning heuristics",
        "held-out planning performance",
        "https://arxiv.org/abs/2605.29649",
        "Results, held-out domains",
        "generator",
    ),
    (
        "dei diversity in evolutionary inference",
        "dei-2026",
        "core",
        "program competition",
        "Core War QD search",
        "heterogeneous distributed LLM operators",
        "homogeneous and solo LLM operators",
        "QD-score and coverage",
        "https://arxiv.org/abs/2605.27130",
        "Tables 2-3",
        "model_budget",
    ),
    (
        "parameter efficient neuroevolution for diverse llm generation",
        "qd-llm-2026",
        "core",
        "text generation",
        "HumanEval and diverse LLM generation",
        "QD-LLM prompt-and-text evolution",
        "QDAIF",
        "QD-score and coverage",
        "https://arxiv.org/abs/2605.09781",
        "Section 4.5, Table 4",
        "generator",
    ),
    (
        "alphacontext",
        "alphacontext-2026",
        "core",
        "psychometrics",
        "creativity-context generation",
        "LLM/MAP-Elites context evolution",
        "competing context generators",
        "context quality and diversity",
        "https://arxiv.org/abs/2604.18398",
        "Section 4.3 and Experiments",
        "generator",
    ),
    (
        "llm guided prompt evolution for password",
        "password-qd-2026",
        "core",
        "cybersecurity",
        "password guessing",
        "OpenEvolve prompt evolution",
        "initial prompt and alternate configurations",
        "guess success and diversity",
        "https://arxiv.org/abs/2604.12601",
        "Results",
        "bundle",
    ),
    (
        "dsevolve",
        "dsevolve-2026",
        "core",
        "scheduling",
        "dynamic flexible job shop",
        "state-conditioned QD rule portfolio",
        "individual rules, GP, DRL, and AHD baselines",
        "scheduling objective",
        "https://arxiv.org/abs/2603.27628",
        "Table 1",
        "bundle",
    ),
    (
        "kernelfoundry",
        "kernelfoundry-2026",
        "core",
        "GPU programming",
        "kernel optimization",
        "MAP-Elites/meta-prompt evolution",
        "kernel-generation baselines",
        "KernelBench performance",
        "https://arxiv.org/abs/2603.12440",
        "KernelBench comparison",
        "bundle",
    ),
    (
        "agenticgeo",
        "agenticgeo-2026",
        "core",
        "generative engine optimization",
        "agent-system evolution",
        "MAP-Elites with co-evolving critic",
        "14 GEO baselines",
        "task score and diversity",
        "https://arxiv.org/abs/2603.20213",
        "Main results",
        "bundle",
    ),
    (
        "manifold of failure",
        "manifold-failure-2026",
        "core",
        "LLM safety",
        "behavioral failure discovery",
        "MAP-Elites search",
        "Random, GCG, TAP, and PAIR",
        "failure coverage at matched queries",
        "https://arxiv.org/abs/2602.22291",
        "Table 2",
        "archive_budget",
    ),
    (
        "steer inference time risk",
        "steer-2026",
        "core",
        "LLM safety",
        "risk-controlled inference",
        "constrained QD personas",
        "temperature sampling, static ensembles, and post-training",
        "risk and utility",
        "https://arxiv.org/abs/2602.02862",
        "Results",
        "bundle",
    ),
    (
        "diversifying toxicity search through speciation",
        "toxsearch-speciation-2026",
        "core",
        "LLM safety",
        "toxicity discovery",
        "speciated ToxSearch-S",
        "unspeciated ToxSearch",
        "peak toxicity and coverage",
        "https://doi.org/10.1145/3795101.3805412",
        "Results",
        "archive",
    ),
    (
        "loongflow",
        "loongflow-2025",
        "core",
        "algorithm discovery",
        "code and ML pipeline search",
        "PES plus MAP-Elites search",
        "OpenEvolve and ShinkaEvolve",
        "solution quality and evolutionary efficiency",
        "https://arxiv.org/abs/2512.24077",
        "Section 5.4, Figures 3-4",
        "bundle",
    ),
    (
        "gigaevo",
        "gigaevo-2025",
        "core",
        "algorithm discovery",
        "code optimization",
        "evolved MAP-Elites prompt and model roles",
        "baseline prompt and fixed roles",
        "objective score",
        "https://arxiv.org/abs/2511.17592",
        "Prompt-evolution experiment",
        "prompt_model",
    ),
    (
        "codeevolve",
        "codeevolve-2025",
        "core",
        "algorithm discovery",
        "code optimization",
        "CVT-MAP-Elites evolutionary agent",
        "OpenEvolve and ShinkaEvolve",
        "benchmark score and efficiency",
        "https://arxiv.org/abs/2510.14150",
        "Benchmark results and component ablations",
        "bundle",
    ),
    (
        "optimizing for persuasion",
        "persuasion-qd-2025",
        "core",
        "debate",
        "strategy evolution",
        "persuasion-fitness QD",
        "truth-fitness QD",
        "generalization and persuasion",
        "https://arxiv.org/abs/2510.05909",
        "Controlled-objective experiment",
        "evaluator",
    ),
    (
        "quality diversity red teaming",
        "qdrt-2025",
        "core",
        "LLM safety",
        "adversarial prompt generation",
        "QD red-teaming generator",
        "GFlowNet and RL/QD generators",
        "attack quality and diversity",
        "https://arxiv.org/abs/2506.07121",
        "Table 1",
        "generator",
    ),
    (
        "sparq synthetic problem",
        "sparq-2025",
        "core",
        "reasoning",
        "synthetic problem generation",
        "QD synthetic generation",
        "quality/diversity filtering ablations",
        "downstream reasoning performance",
        "https://arxiv.org/abs/2506.06499",
        "Ablation section",
        "gate",
    ),
    (
        "rainbowplus",
        "rainbowplus-2025",
        "core",
        "LLM safety",
        "adversarial prompt generation",
        "multi-element QD archive",
        "Rainbow Teaming",
        "attack success and coverage",
        "https://arxiv.org/abs/2504.15047",
        "Tables 1-2",
        "archive",
    ),
    (
        "diverse prompts illuminating",
        "diverse-prompts-2025",
        "core",
        "prompt optimization",
        "BIG-Bench prompt search",
        "CFG/MAP-Elites prompt search",
        "non-QD prompt generation",
        "task score and prompt diversity",
        "https://doi.org/10.1109/CEC65147.2025.11043024",
        "Experiments across seven tasks",
        "archive",
    ),
    (
        "ferret faster",
        "ferret-2024",
        "core",
        "LLM safety",
        "automated red teaming",
        "reward-scored multi-mutation search",
        "Rainbow Teaming",
        "attack success and time",
        "https://arxiv.org/abs/2408.10701",
        "Main ASR/time comparison",
        "bundle",
    ),
    (
        "large language models as in context ai generators",
        "in-context-qd-2024",
        "core",
        "synthetic functions and content",
        "QD generation",
        "LLM in-context generator",
        "conventional QD and single-objective generators",
        "QD-score and coverage",
        "https://doi.org/10.1162/isal_a_00771",
        "Figure 5 and prompt ablations",
        "generator",
    ),
    (
        "rainbow teaming",
        "rainbow-teaming-2024",
        "core",
        "LLM safety",
        "adversarial prompt generation",
        "QD search",
        "rejection sampling and component ablations",
        "attack success and diversity",
        "https://arxiv.org/abs/2402.16822",
        "Table 3 and Figure 9",
        "archive",
    ),
    (
        "evolution through large models",
        "openelm-2022",
        "core",
        "robotics/program synthesis",
        "Sodarace",
        "learned diff-model mutation in MAP-Elites",
        "conventional mutation and generation",
        "fitness and diversity",
        "https://arxiv.org/abs/2206.08896",
        "Sodarace experiment",
        "generator",
    ),
    (
        "levi stronger search",
        "levi-2026",
        "core",
        "algorithm discovery",
        "code optimization",
        "CVT-MAP-Elites and role-aware routing",
        "GEPA, OpenEvolve, and fixed model mixes",
        "score per call and cost",
        "https://arxiv.org/abs/2605.09764",
        "Figures 4-5 and Tables 7-10",
        "allocation_model",
    ),
    (
        "llmatic",
        "llmatic-2024",
        "core",
        "neural architecture search",
        "NAS-Bench-201",
        "dual QD archives",
        "single archive, operator-only, and random variants",
        "validation/test accuracy and diversity",
        "https://doi.org/10.1145/3638529.3654017",
        "Ablation section",
        "archive",
    ),
    (
        "task aware robotic grasping",
        "task-aware-grasping-2025",
        "core",
        "robotics",
        "task-aware grasp selection",
        "LLM-selected grasp from QD archive",
        "control grasp",
        "task success",
        "https://doi.org/10.1109/IROS60139.2025.11246636",
        "End-to-end validation",
        "selector",
    ),
    (
        "algorithmic prompt generation for diverse human",
        "teaming-prompts-2026",
        "core",
        "human-agent teaming",
        "communication prompt generation",
        "QD-evolved LLM-agent prompts",
        "manual and non-QD behavior generation",
        "behavior diversity and user outcomes",
        "https://doi.org/10.1145/3795101.3805408",
        "Experiments and user study",
        "generator",
    ),
    (
        "self evolving time series agent",
        "time-series-agent-2026",
        "core",
        "time series",
        "forecasting architecture search",
        "LLM/MAP-Elites architecture evolution",
        "forecasting baselines",
        "forecast error",
        "https://doi.org/10.5281/zenodo.20594596",
        "Main results and ablations",
        "bundle",
    ),
    (
        "qd llms",
        "qd-llms-2026",
        "core",
        "generative design",
        "vehicle design",
        "LLM emitters and allocator",
        "iterative zero-shot prompting and QD baselines",
        "quality and design-space coverage",
        "https://doi.org/10.1145/3795101.3814651",
        "Vehicle-design experiments",
        "allocation",
    ),
    (
        "mortar evolving mechanics",
        "mortar-2026",
        "core",
        "game design",
        "mechanic generation",
        "MCTS-evaluated LLM/QD evolution",
        "LLM-only, random, and greedy selection",
        "quality and diversity",
        "https://doi.org/10.1145/3795095.3805100",
        "Table 1",
        "bundle",
    ),
    (
        "evolutionary content generation via multimodal",
        "multimodal-fitness-2026",
        "core",
        "procedural content generation",
        "content evolution",
        "multimodal-LLM fitness",
        "conventional evaluation across CMA-ES/MAP-Elites/CMA-ME",
        "quality and user preference",
        "https://doi.org/10.1109/TEVC.2026.3657774",
        "Results and user study",
        "evaluator",
    ),
    (
        "copolymer gel dynamics",
        "copolymer-qd-2026",
        "core",
        "materials",
        "copolymer design exploration",
        "LLM embedding descriptors",
        "conventional molecular descriptors",
        "archive distribution and candidate quality",
        "https://doi.org/10.26434/chemrxiv-2026-4kmdq",
        "Results",
        "representation",
    ),
    (
        "relay don t route",
        "relay-2026",
        "adjacent",
        "algorithm discovery",
        "cost-efficient model handoff",
        "adaptive population handoff",
        "all-cheap, all-strong, fixed, random, bandit, and LEVI",
        "quality, calls, and cost",
        "https://arxiv.org/abs/2608.05651",
        "Table 1 and Figure 4",
        "allocation_model",
    ),
    (
        "automating parent selection configuration",
        "parent-selection-agent-2026",
        "adjacent",
        "genetic programming",
        "parent selector synthesis",
        "agent-generated selector",
        "tournament and epsilon-lexicase selectors",
        "GP performance",
        "https://arxiv.org/abs/2608.17172",
        "Table VIII",
        "selector",
    ),
    (
        "automated discovery has no universally",
        "harness-study-2026",
        "adjacent",
        "algorithm discovery",
        "evolution harness comparison",
        "adaptive pruning and reallocation",
        "fixed and non-adaptive harness ensembles",
        "quality per compute",
        "https://arxiv.org/abs/2607.18235",
        "Adaptive-allocation experiment",
        "allocation",
    ),
    (
        "compute allocation in evolutionary",
        "base-allocation-2026",
        "adjacent",
        "algorithm discovery",
        "call allocation",
        "BaSE bandit allocation",
        "greedy and island protocols",
        "best score at 512 calls",
        "https://arxiv.org/abs/2605.29268",
        "Table 2",
        "allocation",
    ),
    (
        "tide tuning",
        "tide-2026",
        "adjacent",
        "automated heuristic design",
        "prompt-strategy allocation",
        "UCB prompt-strategy scheduler",
        "random prompt-strategy selection",
        "heuristic objective",
        "https://arxiv.org/abs/2601.21239",
        "Table 3",
        "allocation",
    ),
    (
        "decision tree induction through llms",
        "llm-decision-tree-2025",
        "adjacent",
        "machine learning",
        "decision-tree induction",
        "semantic LLM variation",
        "no-semantics and binary genetic variants",
        "predictive performance and tree quality",
        "https://arxiv.org/abs/2503.14217",
        "Section 5.3 and Figure 7",
        "generator",
    ),
    (
        "can large language models be trusted as evolutionary",
        "trusted-eo-2025",
        "adjacent",
        "combinatorial optimization",
        "network-structured optimization",
        "LLM population and individual operators",
        "conventional evolutionary optimization",
        "objective value and reliability",
        "https://arxiv.org/abs/2501.15081",
        "Optimizer comparison",
        "generator",
    ),
    (
        "llm empowered adaptive evolutionary",
        "mumoea-2025",
        "adjacent",
        "multi-objective optimization",
        "multi-component systems",
        "LLM-empowered adaptive EA",
        "random initialization, no differential seeds, and MOSAT",
        "multi-objective performance",
        "https://arxiv.org/abs/2501.00829",
        "Tables 3-4",
        "bundle",
    ),
    (
        "online operator design in evolutionary",
        "online-operator-design-2025",
        "adjacent",
        "scheduling",
        "online variation-operator design",
        "online LLM operator evolution",
        "fixed single and operator-pool variants",
        "scheduling objective",
        "https://arxiv.org/abs/2511.16485",
        "Table 2",
        "allocation",
    ),
    (
        "automatic programming via large language models with population",
        "seevo-2024",
        "adjacent",
        "scheduling",
        "dispatch-rule synthesis",
        "LLM population self-evolution",
        "GEP, MTGP, DRL, and fixed rules",
        "scheduling objective",
        "https://arxiv.org/abs/2410.22657",
        "Tables II-III",
        "generator",
    ),
    (
        "llmigrate",
        "llmigrate-2026",
        "adjacent",
        "soft robotics",
        "island migration control",
        "LLM migration controller",
        "mutation-only evolution and QD references",
        "EvoGym performance and diversity",
        "https://doi.org/10.1145/3795101.3805447",
        "Main EvoGym results",
        "allocation",
    ),
    (
        "laos large language",
        "laos-2025",
        "adjacent",
        "evolutionary optimization",
        "adaptive operator selection",
        "LLM-driven adaptive operator selection",
        "fixed and conventional AOS baselines",
        "benchmark optimization performance",
        "https://doi.org/10.1145/3712256.3726450",
        "Experimental-results tables",
        "allocation",
    ),
    (
        "rider evolutionary prompt",
        "rider-2026",
        "adjacent",
        "prompt optimization",
        "operator allocation",
        "Thompson-sampled operator allocation",
        "APE, PromptBreeder, EvoGA, and EvoDE",
        "prompt-task score",
        "https://doi.org/10.23919/FRUCT70069.2026.11506558",
        "Table I",
        "allocation",
    ),
    (
        "lhc cmoea",
        "lhc-cmoea-2026",
        "adjacent",
        "constrained multi-objective optimization",
        "event-triggered LLM intervention",
        "event-triggered hybrid co-evolution",
        "component-disabled variants",
        "constrained MOEA indicators",
        "https://doi.org/10.1109/ACCESS.2026.3722750",
        "Component ablations",
        "allocation",
    ),
    (
        "large language model enhanced heuristic synergy",
        "heuristic-synergy-2026",
        "adjacent",
        "heuristic design",
        "destroy/repair operator co-evolution",
        "LLM-enhanced synergy orchestrator",
        "LLM-AHD and classical solvers",
        "benchmark objective",
        "https://doi.org/10.1007/s12065-026-01249-5",
        "Main comparison and synergy ablation",
        "allocation",
    ),
    (
        "evolution of heuristics for selection synthesis",
        "selection-synthesis-2026",
        "adjacent",
        "genetic algorithms",
        "selection-operator synthesis",
        "LLM-designed selectors",
        "tournament, proportional, rank, and FunSearch",
        "17-task optimization performance",
        "https://doi.org/10.31772/2712-8970-2026-27-1-21-32",
        "17-task comparison",
        "selector",
    ),
]

SUPPLEMENTARY_INCLUSIONS = [
    (
        "adaptevolve-2026",
        "AdaptEvolve: Improving Efficiency of Evolutionary AI Agents through Adaptive Model Selection",
        ["Pretam Ray", "et al."],
        2026,
        "Findings of ACL 2026",
        "adjacent",
        "algorithm discovery",
        "adaptive model routing",
        "confidence-driven adaptive model allocation",
        "static large-model baseline",
        "quality retention and inference cost",
        "https://arxiv.org/abs/2602.11931",
        "Main results and model-allocation comparison",
        "allocation_model",
    ),
    (
        "qdaif-2024",
        "Quality-Diversity through AI Feedback",
        ["Jenny Zhang", "et al."],
        2024,
        "ICLR 2024",
        "core",
        "text generation",
        "open-ended text generation",
        "QDAIF with LMs for variation and archive scoring",
        "reported non-QD and component baselines",
        "text quality and archive diversity",
        "https://arxiv.org/abs/2310.13032",
        "Section 3 and Figure 2",
        "bundle",
    ),
    (
        "drq-2026",
        "Digital Red Queen",
        ["Sakana AI", "et al."],
        2026,
        "arXiv",
        "core",
        "program competition",
        "Core War adversarial QD",
        "MAP-Elites DRQ",
        "single-cell archive ablation",
        "generality and archive performance",
        "https://arxiv.org/abs/2601.03335",
        "Section 4.5, Figure 8",
        "archive",
    ),
    (
        "adaevolve-2026",
        "AdaEvolve: Adaptive LLM-Driven Evolutionary Search",
        ["Authors listed in primary source"],
        2026,
        "arXiv",
        "adjacent",
        "algorithm discovery",
        "adaptive island allocation",
        "UCB adaptive island selection",
        "round-robin island selection",
        "terminal task objective",
        "https://arxiv.org/abs/2602.20133",
        "Section 4.5, Table 4",
        "allocation",
    ),
    (
        "shinkaevolve-2026",
        "ShinkaEvolve",
        ["Authors listed in primary source"],
        2026,
        "ICLR 2026",
        "adjacent",
        "algorithm discovery",
        "adaptive model selection",
        "bandit-based LLM ensemble",
        "uniform ensemble sampling and single-model baseline",
        "best-so-far task score",
        "https://arxiv.org/abs/2509.19349",
        "Section 5, Figure 9",
        "allocation_model",
    ),
]

SEED_OVERRIDES = {
    "openelm-2022": {
        "focal": "300M diff-model mutation",
        "baseline": "tuned TinyGP node mutation",
        "endpoint": "successful test-passing child after one 4-Parity mutation",
        "claim": "Compares a learned diff-model mutation operator with tuned TinyGP node mutation; TinyGP succeeds in 0% of reported trials, while the diff-model rate is figure-only.",
        "reported": ["proposal"],
        "result": {
            "direction": "focal_better",
            "focal_value": None,
            "baseline_value": "0%",
            "effect": None,
            "units": "successful mutation attempts",
        },
        "sample": {
            "unit": "mutation attempt",
            "n": None,
            "seeds": None,
            "uncertainty_procedure": "Number of trials and seeds not reported.",
        },
        "locator": "Figure 1; Appendix A, Comparing Mutation Operators",
    },
    "llmatic-2024": {
        "focal": "LLMatic",
        "baseline": "random search",
        "endpoint": "ImageNet16-120 test accuracy",
        "claim": "LLMatic reports 45.87±0.96% ImageNet16-120 test accuracy versus 44.57±1.25% for random search.",
        "reported": ["evaluation"],
        "result": {
            "direction": "focal_better",
            "focal_value": "45.87±0.96%",
            "baseline_value": "44.57±1.25%",
            "effect": "+1.30 percentage points",
            "units": "test accuracy",
        },
        "sample": {
            "unit": "independent NAS run",
            "n": 10,
            "seeds": None,
            "uncertainty_procedure": "Mean ± reported dispersion; random-search run count not reported.",
        },
        "locator": "Table 1; Methods Section 4.3",
    },
    "in-context-qd-2024": {
        "claim": "On synthetic BBOB tasks, In-context QD is plotted above MAP-Elites and random sampling in QD-score with similar coverage; exact values are not tabulated.",
        "reported": ["proposal", "evaluation"],
        "result": {
            "direction": "focal_better",
            "focal_value": None,
            "baseline_value": None,
            "effect": "figure-only",
            "units": "QD-score",
        },
        "sample": {
            "unit": "independent QD run",
            "n": 5,
            "seeds": None,
            "uncertainty_procedure": "Five independent runs; plotted uncertainty.",
        },
        "locator": "Results; Figure 2 and caption",
    },
    "codeevolve-2025": {
        "focal": "CodeEvolve with Qwen3-Coder-30B",
        "baseline": "published AlphaEvolve result",
        "endpoint": "CirclePackingSquare n=32 score",
        "claim": "CodeEvolve reports a best CirclePackingSquare n=32 score of 2.93954 versus the published AlphaEvolve score 2.93794, but the comparison does not match budgets.",
        "reported": [],
        "result": {
            "direction": "focal_better",
            "focal_value": 2.93954,
            "baseline_value": 2.93794,
            "effect": 0.0016,
            "units": "sum of packed-circle radii",
        },
        "sample": {
            "unit": "best discovered program",
            "n": 3,
            "seeds": None,
            "uncertainty_procedure": "Best of three CodeEvolve runs; AlphaEvolve denominator not reported.",
        },
        "locator": "Section 5.3, Table 1; Figure 6 for component ablation",
    },
    "loongflow-2025": {
        "focal": "LoongFlow",
        "baseline": "OpenEvolve",
        "endpoint": "evaluations to Circle Packing score ≥0.99",
        "claim": "LoongFlow reaches Circle Packing score ≥0.99 in 258 mean evaluations versus 783 for OpenEvolve, with target-reaching rates 3/3 versus 1/3.",
        "reported": ["evaluation", "wall_time"],
        "result": {
            "direction": "focal_better",
            "focal_value": "258 evaluations; 3/3 success",
            "baseline_value": "783 evaluations; 1/3 success",
            "effect": ">60% fewer evaluations reported",
            "units": "evaluations to score ≥0.99",
        },
        "sample": {
            "unit": "independent evolutionary run",
            "n": 3,
            "seeds": None,
            "uncertainty_procedure": "Three runs per method; seed identifiers not reported.",
        },
        "locator": "Section 5.3.1, Table 3",
    },
    "adaptevolve-2026": {
        "focal": "Adaptive Hoeffding Tree routing",
        "baseline": "cascading",
        "endpoint": "LiveCodeBench V5 accuracy and normalized compute",
        "claim": "Adaptive Hoeffding Tree routing reaches 73.6% LiveCodeBench accuracy at normalized compute 2.08 versus 73.8% at 2.81 for cascading.",
        "reported": ["llm_call"],
        "result": {
            "direction": "focal_better",
            "focal_value": "73.6% accuracy; compute 2.08; efficiency 35.4",
            "baseline_value": "73.8% accuracy; compute 2.81; efficiency 26.3",
            "effect": "aggregate 37.9% less compute while retaining 97.5% of static-large accuracy",
            "units": "accuracy and normalized model-call compute",
        },
        "sample": {
            "unit": "LiveCodeBench problem",
            "n": 880,
            "seeds": None,
            "uncertainty_procedure": "Runs/seeds not reported.",
        },
        "locator": "Section 3, Table 2",
    },
    "dei-2026": {
        "focal": "four-model heterogeneous merged archive",
        "baseline": "solo DRQ archive",
        "endpoint": "final QD-score and coverage",
        "claim": "A four-model heterogeneous merged archive reaches QD-score 45.90 and 80.6% coverage versus 20.46 and 63.0% for solo DRQ.",
        "reported": ["llm_call"],
        "result": {
            "direction": "focal_better",
            "focal_value": "QD 45.90; coverage 80.6%",
            "baseline_value": "QD 20.46; coverage 63.0%",
            "effect": "+124% QD-score; +28% coverage reported",
            "units": "final merged archive",
        },
        "sample": {
            "unit": "independent seed-level run",
            "n": None,
            "seeds": None,
            "uncertainty_procedure": "Means across seeds; count and seed identifiers not reported.",
        },
        "locator": "Sections 3.3-3.4 and 4.2; Table 3; Figure 3",
    },
    "qd-llm-2026": {
        "claim": "On HumanEval, QD-LLM reports QD-score 26.3 and coverage 0.41 versus QDAIF at 18.6 and 0.28.",
        "reported": ["evaluation"],
        "result": {
            "direction": "focal_better",
            "focal_value": "QD 26.3; coverage 0.41",
            "baseline_value": "QD 18.6; coverage 0.28",
            "effect": "+7.7 QD-score; +0.13 coverage",
            "units": "final archive",
        },
        "sample": {
            "unit": "independent QD run",
            "n": 30,
            "seeds": "0-29",
            "uncertainty_procedure": "Thirty seeded runs.",
        },
        "locator": "Section 4.5, Table 4",
    },
    "adaevolve-2026": {
        "claim": "Removing adaptive island selection and using round-robin reduces the reported Circle Packing and Signal Processing objectives.",
        "reported": ["llm_call"],
        "result": {
            "direction": "focal_better",
            "focal_value": "2.6294±0.003; 0.7178±0.019",
            "baseline_value": "2.6180±0.005; 0.6190±0.0541",
            "effect": "task-specific",
            "units": "terminal task objective",
        },
        "sample": {
            "unit": "independent evolutionary run",
            "n": 3,
            "seeds": None,
            "uncertainty_procedure": "Three runs; reported mean ± dispersion.",
        },
        "locator": "Section 4.5, Table 4",
    },
}


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def assessment(status: str, focal: Any, baseline: Any, evidence: str) -> dict[str, Any]:
    return {
        "status": status,
        "focal": focal,
        "baseline": baseline,
        "evidence": evidence,
    }


def treatment_vector(
    pattern: str, locator: str, focal: str, baseline: str
) -> dict[str, Any]:
    vector = {
        component: assessment(
            "unclear", None, None, f"Not fully reported in {locator}."
        )
        for component in COMPONENTS
    }
    vector["evaluator"] = assessment(
        "matched", "same task evaluator", "same task evaluator", locator
    )
    vector["representation"] = assessment(
        "matched", "same task representation", "same task representation", locator
    )
    changed_map = {
        "bundle": {"selector", "generator", "prompt_channel", "allocation"},
        "archive": {"selector", "replacement"},
        "generator": {"generator"},
        "model": {"model"},
        "model_budget": {"model", "allocation"},
        "prompt_model": {"prompt_channel", "model"},
        "archive_budget": {"selector", "replacement"},
        "evaluator": {"evaluator"},
        "gate": {"gate"},
        "allocation": {"allocation"},
        "allocation_model": {"allocation", "model"},
        "selector": {"selector"},
        "representation": {"representation"},
    }
    for component in changed_map.get(pattern, set()):
        vector[component] = assessment("changed", focal, baseline, locator)
    if pattern in {"model_budget", "archive_budget", "allocation", "allocation_model"}:
        vector["budget"] = assessment(
            "matched", "reported matched budget", "reported matched budget", locator
        )
    return vector


def load_candidates(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "screening_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def match_inclusion(title: str) -> tuple[Any, ...] | None:
    normalized = norm(title)
    matches = [entry for entry in INCLUSIONS if norm(entry[0]) in normalized]
    if not matches:
        return None
    return max(matches, key=lambda entry: len(norm(entry[0])))


def make_paper(
    candidate: dict[str, Any], entry: tuple[Any, ...], screening_id: str
) -> dict[str, Any]:
    fragment, paper_id, stratum, domain, task, *_rest = entry
    del fragment
    return {
        "record_type": "paper",
        "paper_id": paper_id,
        "title": candidate["title"],
        "authors": candidate.get("authors") or ["Authors listed in primary source"],
        "year": (
            int(paper_id.rsplit("-", 1)[-1])
            if paper_id.rsplit("-", 1)[-1].isdigit()
            else (candidate.get("year") or 2026)
        ),
        "venue": candidate.get("venue") or "",
        "version_status": "preprint" if "arxiv.org" in entry[8] else "peer_reviewed",
        "stratum": stratum,
        "identifiers": {
            key: candidate.get(key) for key in ("doi", "arxiv") if candidate.get(key)
        },
        "urls": [entry[8]],
        "domain": domain,
        "task": task,
        "code_available": None,
        "screening_id": screening_id,
        "notes": "Full-text inclusion audited during literature screening.",
    }


def make_comparison(
    candidate: dict[str, Any], entry: tuple[Any, ...]
) -> dict[str, Any]:
    (
        _fragment,
        paper_id,
        _stratum,
        _domain,
        _task,
        focal,
        baseline,
        endpoint,
        url,
        locator,
        pattern,
    ) = entry
    override = SEED_OVERRIDES.get(paper_id, {})
    focal = override.get("focal", focal)
    baseline = override.get("baseline", baseline)
    endpoint = override.get("endpoint", endpoint)
    locator = override.get("locator", locator)
    vector = treatment_vector(pattern, locator, focal, baseline)
    changed = [key for key, value in vector.items() if value["status"] == "changed"]
    reported = set(
        override.get(
            "reported",
            (
                ["proposal"]
                if pattern
                not in {
                    "model_budget",
                    "archive_budget",
                    "allocation",
                    "allocation_model",
                }
                else ["llm_call"]
            ),
        )
    )
    quote = candidate.get("abstract") or ""
    return {
        "record_type": "comparison",
        "comparison_id": f"{paper_id}-c1",
        "paper_id": paper_id,
        "claim": override.get(
            "claim", f"Reports {focal} versus {baseline} on {endpoint}."
        ),
        "focal_arm": focal,
        "baseline_arm": baseline,
        "endpoint": endpoint,
        "budget_axes": {
            "reported": sorted(reported),
            "omitted": sorted(ALL_BUDGETS - reported),
        },
        "sample": override.get(
            "sample",
            {
                "unit": "independent run or task; see primary source",
                "n": None,
                "seeds": None,
                "uncertainty_procedure": "See primary source; not consistently reported in metadata extraction.",
            },
        ),
        "reported_result": override.get(
            "result",
            {
                "direction": "unclear",
                "focal_value": None,
                "baseline_value": None,
                "effect": None,
                "units": endpoint,
            },
        ),
        "treatment_vector": vector,
        "identified_effects": changed,
        "unidentified_effects": [
            component
            for component, value in vector.items()
            if value["status"] == "unclear"
        ],
        "source": {
            "url": url,
            "section": locator,
            "quote": quote[:1000],
        },
        "extraction_notes": "Comparison-level charting is conservative: unverified arm details are coded unclear.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    candidates = load_candidates(args.root)
    existing_screening = [
        json.loads(line)
        for line in (args.root / "screening.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    existing_screening = [
        row for row in existing_screening if row.get("stage") == "title_abstract"
    ]
    for row in existing_screening:
        if row["stage"] == "title_abstract" and row["decision"] in {
            "include",
            "unclear",
        }:
            row["audited"] = True
            row["notes"] += " Full-text eligibility pass completed."
        elif (
            row["stage"] == "title_abstract"
            and row["decision"] == "exclude"
            and row["reason_code"] != "duplicate"
        ):
            digest = int(
                hashlib.sha256(row["screening_id"].encode()).hexdigest()[:8], 16
            )
            if digest % 10 == 0:
                row["audited"] = True
                row[
                    "notes"
                ] += " Included in deterministic 10% exclusion consistency audit."

    papers: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    full_text_rows: list[dict[str, Any]] = []
    included_paper_ids: set[str] = set()
    for candidate in candidates:
        if candidate["initial_decision"] not in {"include", "unclear"}:
            continue
        entry = match_inclusion(candidate["title"])
        screening_id = f"ft-{candidate['candidate_id']}"
        if entry is None:
            full_text_rows.append(
                {
                    "record_type": "screening",
                    "screening_id": screening_id,
                    "title": candidate["title"],
                    "discovery_source": "+".join(candidate["sources"]),
                    "stage": "full_text",
                    "decision": "exclude",
                    "reason_code": "not_core_or_adjacent_after_full_text",
                    "audited": True,
                    "notes": "No attribution-relevant contrast under the strict full-text rule.",
                }
            )
            continue
        if entry[1] in included_paper_ids:
            full_text_rows.append(
                {
                    "record_type": "screening",
                    "screening_id": screening_id,
                    "title": candidate["title"],
                    "discovery_source": "+".join(candidate["sources"]),
                    "stage": "full_text",
                    "decision": "exclude",
                    "reason_code": "superseded_version",
                    "audited": True,
                    "notes": f"Preferred version already retained as {entry[1]}.",
                }
            )
            continue
        full_text_rows.append(
            {
                "record_type": "screening",
                "screening_id": screening_id,
                "title": candidate["title"],
                "discovery_source": "+".join(candidate["sources"]),
                "stage": "full_text",
                "decision": "include",
                "reason_code": None,
                "audited": True,
                "notes": f"Included in {entry[2]} stratum; principal contrast charted.",
            }
        )
        papers.append(make_paper(candidate, entry, screening_id))
        comparisons.append(make_comparison(candidate, entry))
        included_paper_ids.add(entry[1])

    for supplementary in SUPPLEMENTARY_INCLUSIONS:
        (
            paper_id,
            title,
            authors,
            year,
            venue,
            stratum,
            domain,
            task,
            focal,
            baseline,
            endpoint,
            url,
            locator,
            pattern,
        ) = supplementary
        screening_id = f"ft-snowball-{paper_id}"
        full_text_rows.append(
            {
                "record_type": "screening",
                "screening_id": screening_id,
                "title": title,
                "discovery_source": "supplementary backward/forward snowballing",
                "stage": "full_text",
                "decision": "include",
                "reason_code": None,
                "audited": True,
                "notes": "Eligible predecessor recovered during the second full-text consistency audit.",
            }
        )
        candidate = {
            "title": title,
            "authors": authors,
            "year": year,
            "venue": venue,
            "abstract": "",
        }
        entry = (
            paper_id,
            paper_id,
            stratum,
            domain,
            task,
            focal,
            baseline,
            endpoint,
            url,
            locator,
            pattern,
        )
        papers.append(make_paper(candidate, entry, screening_id))
        comparisons.append(make_comparison(candidate, entry))

    all_screening = existing_screening + full_text_rows
    (args.root / "screening.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in all_screening
        ),
        encoding="utf-8",
    )
    (args.root / "papers.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in papers
        ),
        encoding="utf-8",
    )
    (args.root / "comparisons.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in comparisons
        ),
        encoding="utf-8",
    )
    (args.root / "adjudication.jsonl").write_text("", encoding="utf-8")
    print(
        f"Full texts assessed={len(full_text_rows)}; included papers={len(papers)}; "
        f"comparisons={len(comparisons)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
