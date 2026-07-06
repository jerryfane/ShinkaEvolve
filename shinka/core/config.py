from dataclasses import dataclass, field, fields
from typing import List, Optional, Union

from shinka.core.acceptance import AcceptanceGateConfig
from shinka.llm import BanditBase
from shinka.defaults import (
    DEFAULT_TASK_SYS_MSG,
    default_llm_dynamic_selection_kwargs,
    default_llm_kwargs,
    default_llm_models,
    default_patch_type_probs,
    default_patch_types,
    default_prompt_patch_type_probs,
    default_prompt_patch_types,
)

FOLDER_PREFIX = "gen"


@dataclass
class EvolutionConfig:
    task_sys_msg: Optional[str] = DEFAULT_TASK_SYS_MSG
    patch_types: List[str] = field(default_factory=default_patch_types)
    patch_type_probs: List[float] = field(default_factory=default_patch_type_probs)
    num_generations: int = 50
    max_patch_resamples: int = 3
    max_patch_attempts: int = 1
    job_type: str = "local"
    language: str = "python"
    llm_models: List[str] = field(default_factory=default_llm_models)
    llm_dynamic_selection: Optional[Union[str, BanditBase]] = "ucb"
    llm_dynamic_selection_kwargs: dict = field(
        default_factory=default_llm_dynamic_selection_kwargs
    )
    llm_kwargs: dict = field(default_factory=default_llm_kwargs)
    meta_rec_interval: Optional[int] = 10
    meta_llm_models: Optional[List[str]] = None
    meta_llm_kwargs: dict = field(default_factory=lambda: {})
    meta_max_recommendations: int = 5
    sample_single_meta_rec: bool = True
    embedding_model: Optional[str] = "text-embedding-3-small"
    init_program_path: Optional[str] = "initial.py"
    results_dir: Optional[str] = None
    max_novelty_attempts: int = 3
    code_embed_sim_threshold: float = 0.99
    novelty_llm_models: Optional[List[str]] = None
    novelty_llm_kwargs: dict = field(default_factory=lambda: {})
    use_text_feedback: bool = False
    max_api_costs: Optional[float] = None
    inspiration_sort_order: str = "ascending"
    enable_controlled_oversubscription: bool = False
    proposal_target_mode: str = "adaptive"
    proposal_target_min_samples: int = 5
    proposal_target_ratio_cap: float = 2.0
    proposal_buffer_max: int = 2
    proposal_target_hard_cap: Optional[int] = None
    proposal_target_ewma_alpha: float = 0.3

    # Meta-prompt evolution settings.
    evolve_prompts: bool = False
    prompt_patch_types: List[str] = field(default_factory=default_prompt_patch_types)
    prompt_patch_type_probs: List[float] = field(
        default_factory=default_prompt_patch_type_probs
    )
    prompt_evolution_interval: Optional[int] = None
    prompt_archive_size: int = 10
    prompt_llm_models: Optional[List[str]] = None
    prompt_llm_kwargs: dict = field(default_factory=lambda: {})
    prompt_ucb_exploration_constant: float = 1.0
    prompt_epsilon: float = 0.1
    prompt_evo_top_k_programs: int = 3
    prompt_percentile_recompute_interval: int = 20

    # PACE anytime-valid acceptance gate. ``None`` (default) is equivalent to a
    # disabled gate: behaviour is byte-identical to stock ShinkaEvolve. Supply an
    # AcceptanceGateConfig (or a plain mapping from a YAML/Hydra surface, which is
    # coerced in ``__post_init__``) with ``enabled=True`` to activate it.
    acceptance_gate: Optional[AcceptanceGateConfig] = None

    def __post_init__(self) -> None:
        # Coerce a mapping (e.g. a Hydra/OmegaConf DictConfig or a plain dict
        # coming from ``EvolutionConfig(**configs["evo_config"])``) into a typed
        # AcceptanceGateConfig so the runner can rely on attribute access.
        gate = self.acceptance_gate
        if gate is not None and not isinstance(gate, AcceptanceGateConfig):
            gate_kwargs = dict(gate)
            try:
                self.acceptance_gate = AcceptanceGateConfig(**gate_kwargs)
            except TypeError:
                # A raw TypeError from unexpected keyword arguments is opaque;
                # translate it into an actionable message that names the bad
                # keys and the valid ones.
                valid = [f.name for f in fields(AcceptanceGateConfig)]
                bad = sorted(set(gate_kwargs) - set(valid))
                raise ValueError(
                    f"acceptance_gate: unknown keys {bad}; "
                    f"valid keys: {valid}"
                ) from None
