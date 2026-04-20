"""Head-to-head dataset comparison suite.

This package ships the machinery for comparing our RL dataset against one
or more external baselines (e.g. ``obiwan96/endless-terminals``,
``open-thoughts/OpenThoughts-TB-dev``) along five analysis axes:
difficulty, command-mix, composition (our taxonomy projected onto baselines),
diversity, realism, and verifier rigor.

Entry points:

* ``python -m rl_data.comparison.cli`` — run the analysis
* ``python -m rl_data.comparison.adapters.endless_terminals`` — ingest ET
* ``python -m rl_data.comparison.adapters.openthoughts_tb`` — ingest OT
* ``python -m rl_data.comparison.taxonomy_classifier`` — classify external
  tasks into our taxonomy so they can be compared compositionally
"""

from rl_data.comparison.core import DatasetSpec, save_fig_with_data  # noqa: F401

__all__ = ["DatasetSpec", "save_fig_with_data"]
