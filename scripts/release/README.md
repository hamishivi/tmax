# Preserved-thinking release correction

The TMax v1.1 2B and 27B repositories initially shipped templates that differed
from their preserved-thinking evaluations. The 2B template disabled new thinking
by default. Both templates could discard earlier reasoning after a user reprompt,
including a harness format-error recovery message.

`thinking_templates/` contains the exact templates used in those evaluations.
`preserved_thinking_scores.json` records the completed evaluation sweeps and their
template hashes. Under that configuration, 2B selects step 400 and 27B selects
step 200 on TB-Lite. The corrected templates enable thinking by default and retain
reasoning across tool responses and user reprompts; an explicit
`enable_thinking=false` still disables new thinking.

`fix_preserved_thinking.py` prepares and verifies the cards, template behavior,
checkpoint selection, and source model-file inventories before publication.
It commits to all five milestone branches and `main` in each affected repository.
The 27B `main` tree uses the existing step-0200 checkpoint. Git commits retain the
previous branch heads in history and reject concurrent branch changes. Bare Git
transfers the original LFS pointers without downloading or rewriting weight data.
All non-documentation model files are checked against the selected checkpoint.

Run from the repository root:

```bash
uv run --no-project --with 'huggingface_hub>=2.1.1' --with jinja2 \
  python scripts/release/fix_preserved_thinking.py prepare --directory /tmp/tmax-thinking-fix

# Inspect the prepared cards and plan.json before publishing.
uv run --no-project --with 'huggingface_hub>=2.1.1' --with jinja2 \
  python scripts/release/fix_preserved_thinking.py apply --directory /tmp/tmax-thinking-fix

uv run --no-project --with 'huggingface_hub>=2.1.1' --with jinja2 \
  python scripts/release/fix_preserved_thinking.py verify --directory /tmp/tmax-thinking-fix
```

Publication requires an existing authenticated `hamishivi` Hugging Face account.
`apply` can resume using its receipts; it refuses a branch that changed outside
the prepared plan. The checked-in `preserved_thinking_receipts.json` records the
12 published revisions and verified source weight fingerprints. Each affected
Hub branch also includes `preserved_thinking.json` with its template hash and
checkpoint provenance.

The published scores come from the existing three-trial evaluations used by the
paper. This correction runs no new model rollouts. The historical conversations
in [`evaluation_assets/v1.1/`](../../evaluation_assets/v1.1/) retain their original
scores and evaluated checkpoint identities.
