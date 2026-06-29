"""MAP-Elites candidate emitters."""

from .base import CandidateEmitter, EmitterOutput, MapElitesEmitter, strip_seed
from .genetic_emitter import DEFAULT_MUTATION_SCALE, GeneticEmitter
from .genetics import GENOME_SIZE, decode_genome, encode_world
from .archive_neighbors import moore_neighbor_elites
from .llm_emitter import (
    LlmEmitter,
    build_user_prompt,
    format_current_elite_json,
    format_few_shot_block,
)
from .llm_prompts import (
    DEFAULT_SYSTEM_PROMPT_PATH,
    DEFAULT_SYSTEM_PROMPT_PATH_CVT,
    DEFAULT_USER_PROMPT_PATH,
    emitter_prompt_version,
    load_system_prompt_template,
    load_user_prompt_template,
    render_cvt_system_prompt,
    render_system_prompt,
    render_system_prompt_for_archive_type,
    system_prompt_path_for_archive_type,
    system_prompt_version,
    user_prompt_version,
)
from .random_emitter import RandomEmitter
from .stub import StubCandidateEmitter

__all__ = [
    "CandidateEmitter",
    "DEFAULT_MUTATION_SCALE",
    "EmitterOutput",
    "DEFAULT_SYSTEM_PROMPT_PATH",
    "DEFAULT_SYSTEM_PROMPT_PATH_CVT",
    "DEFAULT_USER_PROMPT_PATH",
    "GENOME_SIZE",
    "GeneticEmitter",
    "LlmEmitter",
    "build_user_prompt",
    "MapElitesEmitter",
    "RandomEmitter",
    "StubCandidateEmitter",
    "decode_genome",
    "encode_world",
    "format_current_elite_json",
    "format_few_shot_block",
    "load_system_prompt_template",
    "load_user_prompt_template",
    "moore_neighbor_elites",
    "render_cvt_system_prompt",
    "render_system_prompt",
    "render_system_prompt_for_archive_type",
    "system_prompt_path_for_archive_type",
    "strip_seed",
    "emitter_prompt_version",
    "system_prompt_version",
    "user_prompt_version",
]
