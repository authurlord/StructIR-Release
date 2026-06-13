# Model & Dataset Cards

All models are used off the shelf (frozen encoder + a fine-tuned
cross-encoder reranker); the GNN graph-slot training/build code is in `src/structir_gnn/` (HGTConv +
gated residual + InfoNCE); `signals.py` consumes its exported refinement. As the
paper reports (Sec. 3), the slot is a registered candidate whose deployable
contribution is null on this suite, so no frozen plan activates it. All datasets are public components of the IBM
Table-Text IR suite.

## Models

| Role | Model | Link |
|---|---|---|
| Dense encoder `M_embed` (frozen; dense leg + GNN node init + routing probe) | **bge-m3** | https://huggingface.co/BAAI/bge-m3 |
| Cross-encoder reranker `M_rerank` (fine-tuned on one source, zero-shot elsewhere) | **bge-reranker-v2-m3** | https://huggingface.co/BAAI/bge-reranker-v2-m3 |
| Alternative dense backbone (AIT-QA ablation) | **snowflake-arctic-embed-m-v2.0** | https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0 |
| Supervised table-embedding baseline | **granite-embedding-english-r2** (Granite-R2) | https://huggingface.co/ibm-granite/granite-embedding-english-r2 |
| Sparse-neural baseline | **SPLADE** (cocondenser-ensembledistil) | https://huggingface.co/naver/splade-cocondenser-ensembledistil |
| Graph slot `M_gnn` (training/build code in `src/structir_gnn/`; registered candidate, null deployable contribution on this suite, see paper Sec. 3) | HGTConv + gated residual + InfoNCE | [Hu et al., *Heterogeneous Graph Transformer*, WWW 2020](https://arxiv.org/abs/2003.01332) |
| Agent + reader LLM (offline) | a local 35B Mixture-of-Experts model (Qwen3 MoE family) | https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f |
| Agent + reader LLM (online frontier check) | an OpenAI-compatible online frontier model | set via `STRUCTIR_LLM_BASE` / `STRUCTIR_LLM_MODEL` |

The agent/reader endpoint is any OpenAI-compatible server; the released
code does not bundle or require a specific provider.

## Datasets — IBM Table-Text IR suite

8 datasets across 4 source families. Suite:
the public [IBM Table+Text IR Evaluation collection](https://huggingface.co/collections/ibm-research/table-text-ir-evaluation).

| Dataset | Family | Source paper |
|---|---|---|
| **[OTT-QA](https://huggingface.co/datasets/ibm-research/OTTQASmallRetrieval)** | Wikipedia | [Chen et al., ICLR 2021](https://arxiv.org/abs/2010.10439) |
| **[NQ-Tables](https://huggingface.co/datasets/ibm-research/NQTablesRetrieval)** | Wikipedia | [Herzig et al., NAACL 2021 (DTR)](https://aclanthology.org/2021.naacl-main.43/) |
| **[OpenWikiTables](https://huggingface.co/datasets/ibm-research/OpenWikiTablesRetrieval)** | Wikipedia | [Kweon et al., ACL 2023](https://aclanthology.org/2023.findings-acl.652/) |
| **[FeTaQA](https://huggingface.co/datasets/ibm-research/FeTaQARetrieval)** | Wikipedia | [Nan et al., TACL 2022](https://arxiv.org/abs/2104.00369) |
| **[MultiHierTT](https://huggingface.co/datasets/ibm-research/MultiHierttRetrieval)** | Financial | [Zhao et al., ACL 2022](https://aclanthology.org/2022.acl-long.454/) |
| **[AIT-QA](https://huggingface.co/datasets/ibm-research/AITQARetrieval)** | Financial | [Katsis et al., NAACL 2022](https://arxiv.org/abs/2106.12944) |
| **[StatCan](https://huggingface.co/datasets/ibm-research/StatCanDialogueRetrieval)** | Statistical | [StatCanDialogue (Lu et al., EACL 2023)](https://aclanthology.org/2023.findings-eacl.92/) |
| **[WatsonxDocs](https://huggingface.co/datasets/ibm-research/WatsonxDocsQARetrieval)** | Technical docs | [IBM Table+Text IR collection](https://huggingface.co/datasets/ibm-research/WatsonxDocsQARetrieval) |
