# Preserved-thinking release correction

The TMax v1.1 2B and 27B repositories initially shipped templates that differed
from their preserved-thinking evaluations. The 2B template disabled new thinking
by default. Both templates could discard earlier reasoning after a user reprompt,
including a harness format-error recovery message.

[`thinking_templates/`](thinking_templates/) contains the exact templates used
in those evaluations: [Qwen 3.5](thinking_templates/qwen35-keep-think.jinja) for
2B and [Qwen 3.6](thinking_templates/qwen36-keep-think.jinja) for 27B. They enable
thinking by default and retain reasoning across tool responses and user
reprompts; an explicit `enable_thinking=false` still disables new thinking.

The templates are shipped on all five milestone branches and `main` of
[TMax 2B v1.1](https://huggingface.co/TMaxxx/tmax-2b-v1.1) and
[TMax 27B v1.1](https://huggingface.co/TMaxxx/tmax-27b-v1.1). Under that
configuration, 2B selects step 400 and 27B selects step 200 on TB-Lite. Each Hub
branch includes `release_manifest.json` with its evaluation results and
`preserved_thinking.json` with its template hash and checkpoint provenance.

The published scores come from the existing three-trial evaluations used by the
paper. The historical conversations
in [`evaluation_assets/v1.1/`](../../evaluation_assets/v1.1/) retain their original
scores and evaluated checkpoint identities.
