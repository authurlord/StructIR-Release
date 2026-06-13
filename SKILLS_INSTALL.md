# Installing use-structir for Code Agents

Current StructIR version: 0.2.0 (structir_skill 0.2.0 / structir_agent 0.1.0)

*Format follows the rdblearn skill convention
(github.com/HKUSHXLab/rdblearn → SKILLS_INSTALL.md).*

## Prerequisites

- Claude Code, Codex, OpenCode.ai, or Cursor installed
- Git installed
- Python ≥ 3.10; `pip install rank_bm25 numpy` (minimal, CPU-only).
  Optional GPU stack: `sentence-transformers torch` (dense leg + reranker).

## Installation Steps

### 1. Create the skill folder

`<skill-dir>` is the agent's skills root (without a leading `~/`): `.claude`
for Claude Code, `.codex` for Codex, `.opencode` for OpenCode. Paths below are
relative to your home directory.

```bash
rm -rf ~/<skill-dir>/skills/use-structir
mkdir -p ~/<skill-dir>/skills/use-structir/codes
mkdir -p ~/<skill-dir>/skills/use-structir/docs
```

### 2. Clone StructIR and install

```bash
git clone --depth 1 <ANON_RELEASE_URL> \
    ~/<skill-dir>/skills/use-structir/codes/StructIR-Release
pip install -e ~/<skill-dir>/skills/use-structir/codes/StructIR-Release
```

Verify:

```bash
python -c "import structir_skill; print(structir_skill.__version__)"   # 0.2.0
python ~/<skill-dir>/skills/use-structir/codes/StructIR-Release/examples/single_corpus_smoke.py
```

### 3. Create the Skill

```bash
SIR=~/<skill-dir>/skills/use-structir/codes/StructIR-Release
cp $SIR/skill/SKILL.md          ~/<skill-dir>/skills/use-structir/SKILL.md
cp $SIR/README.md               ~/<skill-dir>/skills/use-structir/docs/structir_README.md
cp -r $SIR/examples             ~/<skill-dir>/skills/use-structir/docs/examples
cp $SIR/docs/agent_framework.md ~/<skill-dir>/skills/use-structir/docs/agent_framework.md
```

### 4. Provide Skill Status

Show available skills and check whether `use-structir` is on the list.

## Troubleshooting

- **Skill not found**: ensure `~/<skill-dir>/skills/use-structir/SKILL.md` exists; use the `skill` tool to list.
- **Every corpus profiles as `docs`**: you passed `id/title/text` only. Load real structural fields via `structir_skill.load_structured_corpus(...)` (see SKILL.md).
- **`dense/gnn unavailable ... running BM25-only`**: the GPU stack is absent; install `sentence-transformers torch` for the dense leg + reranker.
