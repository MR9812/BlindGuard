# BlindGuard

Official implementation of **BlindGuard: Safeguarding LLM-based Multi-Agent Systems under Unknown Attacks** (ACL 2026 Main).

BlindGuard models multi-agent LLM communication as a directed graph and trains graph neural networks (GNNs) on dialogue embeddings to detect anomalous (attacker) agents during multi-turn collaboration.

## Citation

```bibtex
@inproceedings{miao2026blindguard,
  title={Blindguard: Safeguarding llm-based multi-agent systems under unknown attacks},
  author={Miao, Rui and Liu, Yixin and Wang, Yili and Shen, Xu and Tan, Yue and Dai, Yiwei and Pan, Shirui and Wang, Xin},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={39215--39234},
  year={2026}
}
```

## Repository Structure

| Directory | Attack Scenario | Benchmark / Data |
|-----------|-----------------|------------------|
| `PI/` | Prompt injection | MMLU, CSQA, GSM8K |
| `MA/` | Memory attack | MSMARCO (PoisonRAG-style) |
| `MA_CSQA/` | Memory attack | Commonsense QA |
| `TA/` | Tool attack | InjecAgent-style tool misuse |

Each module shares a similar layout: `agents.py`, `train.py` (supervised G-Safeguard), `train_un1.py` / `train_un2.py` (unsupervised defenses), `main_defense_for_different_topology*.py` (evaluation), and `evaluate_output.py` (ASR / AUROC).

Pretrained weights live in `checkpoint/` and `checkpoint_un2/`. Data-generation shell scripts are in `scripts/`.

## Setup

```bash
pip install torch torch-geometric torch-scatter einops sentence-transformers openai scikit-learn tqdm pandas pyarrow

export OPENAI_API_KEY="your_key"
export BASE_URL="your_openai_compatible_base_url"
```

## General Workflow

```
1. Generate multi-agent dialogue data (gen_graph / scripts/*.sh)
2. merge_datasets.py  → dataset.json
3. gen_training_dataset.py  → embedding pickles
4. train / train_un1 / train_un2  → GNN checkpoints
5. main_defense*.py  → result JSON under result/
6. evaluate_output.py  → ASR and AUROC
```

### Graph Topologies

Evaluation runs under four topologies: `random` (adjacency from data), `chain`, `tree`, and `star`.

### Defense Methods

| Method | Train | Evaluate | Notes |
|--------|-------|----------|-------|
| G-Safeguard | `train.py` | `main_defense_for_different_topology.py` | Supervised |
| Dominant | `train_un2.py` | `main_defense_for_different_topology1.py` | Graph autoencoder |
| TAM | `train_un2.py` | `main_defense_for_different_topology1.py` | TAM anomaly detection |
| PREM | `train_un2.py` | `main_defense_for_different_topology1.py` | PREM-GAD |
| BlindGuard | `train_un1.py` | `main_defense_for_different_topology1.py` | Contrastive learning |

---

## PI — Prompt Injection (MMLU / CSQA / GSM8K)

Run all commands from `PI/`.

### 1. Data Generation

**Train dialogues:**

```bash
cd PI
bash scripts/train/gen_conversation_train_mmlu.sh
bash scripts/train/gen_conversation_train_csqa.sh
bash scripts/train/gen_conversation_train_gsm8k.sh
```

**Test dialogues:**

```bash
bash scripts/test/gen_conversation_test_mmlu.sh
bash scripts/test/gen_conversation_test_csqa.sh
bash scripts/test/gen_conversation_test_gsm8k.sh
```

**Merge:**

```bash
python merge_datasets.py --dataset mmlu --phase train
python merge_datasets.py --dataset mmlu --phase test
# Repeat for csqa and gsm8k
```

**Embeddings:**

```bash
# Unsupervised training
python gen_training_dataset.py --dataset mmlu --type unsuper
python gen_training_dataset.py --dataset csqa --type unsuper
python gen_training_dataset.py --dataset gsm8k --type unsuper

# Supervised G-Safeguard (copy train → train1 first)
python gen_training_dataset.py --dataset mmlu --type super
# Repeat for csqa / gsm8k

# Test embeddings (BlindGuard validation)
python gen_training_dataset.py --dataset mmlu --type test
python gen_training_dataset.py --dataset csqa --type test
python gen_training_dataset.py --dataset gsm8k --type test
```

| `--type` | Input | Output |
|----------|-------|--------|
| `unsuper` | `train/dataset.json` | `ModelTrainingSet/{dataset}/dataset.pkl` |
| `super` | `train1/dataset.json` | `ModelTrainingSet/{dataset}/dataset1.pkl` |
| `test` | `test/dataset.json` | `ModelTrainingSet/{dataset}/test_dataset.pkl` |

### 2. Training

```bash
# G-Safeguard
python train.py --dataset mmlu --epochs 50 --batch_size 32 --lr 0.001
python train.py --dataset csqa --epochs 50 --batch_size 32 --lr 0.001
python train.py --dataset gsm8k --epochs 50 --batch_size 32 --lr 0.001

# Dominant / TAM / PREM
python train_un2.py --dataset mmlu --epochs 50 --batch_size 32 --lr 0.001 --defend_type Dominant
python train_un2.py --dataset mmlu --epochs 50 --batch_size 32 --lr 0.001 --defend_type TAM
python train_un2.py --dataset mmlu --epochs 50 --batch_size 32 --lr 0.001 --defend_type PREM
# Repeat --dataset csqa / gsm8k

# BlindGuard
python train_un1.py --dataset mmlu --epochs 100 --batch_size 1 --lr 0.001 --defend_type SCL --weight_decay 1e-4
python train_un1.py --dataset csqa --epochs 20 --batch_size 1 --lr 0.001 --defend_type SCL --weight_decay 1e-4
python train_un1.py --dataset gsm8k --epochs 100 --batch_size 1 --lr 0.001 --defend_type SCL --weight_decay 5e-5
```

Checkpoints are saved under `./checkpoint/{dataset}/` and `./checkpoint_un2/{dataset}/`.

### 3. Evaluation

Common flags: `--samples 60 --topk 3 --model_type gpt-4o-mini`. Results go to `./result/{dataset}/{graph_type}/`.

**G-Safeguard** — `main_defense_for_different_topology.py` (GSM8K: `main_defense_for_different_topology_gsm8k.py`):

```bash
CKPT=./checkpoint/mmlu/20250701_210222-defend_type_Gsafe_hiddim_1024-heads_8-layers_2-epochs_50-lr_0.001-dropout_0.2-wd_0.0002.pth

python main_defense_for_different_topology.py \
  --graph_type random --gnn_checkpoint_path $CKPT \
  --dataset mmlu --samples 60 --topk 3 --get_no_defense True --model_type gpt-4o-mini
# Repeat for graph_type: chain, tree, star
```

**Dominant / TAM / PREM / BlindGuard** — `main_defense_for_different_topology1.py` (GSM8K: `main_defense_for_different_topology1_gsm8k.py`):

```bash
CKPT=./checkpoint_un2/mmlu/20250701_173128-defend_type_Dominant-hiddim_1024-latent_512-heads_8-layers_2-epochs_50-lr_0.001-temp_0.1.pth

python main_defense_for_different_topology1.py \
  --graph_type random --gnn_checkpoint_path $CKPT \
  --dataset mmlu --samples 60 --defend_type Dominant --topk 3 --model_type gpt-4o-mini
# Repeat for chain / tree / star and other defend_type values
```

---

## MA — Memory Attack

Run from `MA/`.

### Data

```bash
cd MA
bash scripts/train/gen_conversation_train.sh
bash scripts/test/gen_conversation_test.sh

python merge_datasets.py --phase train
python merge_datasets.py --phase test

python gen_training_dataset.py --type unsuper
python gen_training_dataset.py --type super
python gen_training_dataset.py --type test
```

### Training

```bash
python train.py --epochs 50 --batch_size 32 --lr 0.001
python train_un2.py --epochs 50 --batch_size 32 --lr 0.001 --defend_type Dominant
python train_un2.py --epochs 50 --batch_size 32 --lr 0.001 --defend_type TAM
python train_un2.py --epochs 50 --batch_size 32 --lr 0.001 --defend_type PREM
python train_un1.py --epochs 10 --batch_size 1 --lr 0.001 --defend_type SCL
```

### Evaluation

```bash
CKPT=./checkpoint/memory_attack/20250705_174745-hiddim_1024-heads_8-layers_2-epochs_50-lr_0.001-dropout_0.2-wd_0.0002.pth

python main_defense_for_different_topology.py \
  --graph_type random --gnn_checkpoint_path $CKPT \
  --topk 3 --get_no_defense True --model_type gpt-4o-mini
# Repeat for chain / star / tree

# Unsupervised methods: main_defense_for_different_topology1.py --defend_type {Dominant|TAM|PREM|SCL}
```

Results: `./result/memory_attack/{graph_type}/`.

---

## MA_CSQA — Memory Attack on Commonsense QA

Run from `MA_CSQA/`. Unsupervised training uses **benign** dialogues (`num_attackers=0`).

### Data

```bash
cd MA_CSQA
python gen_graph.py --num_nodes 8 --sparsity 0.2 --num_graphs 20 --num_attackers 0 --samples 40 --model_type gpt-4o-mini --phase train
# Repeat for sparsity 0.4, 0.6, 0.8, 1.0

python merge_datasets.py --phase train
python gen_training_dataset.py
```

Generate test data with `scripts/test/gen_conversation_test.sh` and `merge_datasets.py --phase test`.

### Training & Evaluation

Same commands as `MA/` (train / train_un1 / train_un2 / main_defense*). Example G-Safeguard checkpoint: `checkpoint/memory_attack/20250727_090959-...`.

---

## TA — Tool Attack

Run from `TA/`.

```bash
cd TA
bash scripts/train/gen_conversation_train.sh && python merge_datasets.py --phase train
bash scripts/test/gen_conversation_test.sh && python merge_datasets.py --phase test

python gen_training_dataset.py
python train.py --epochs 50 --batch_size 32 --lr 0.001
python train_un1.py --epochs 50 --batch_size 1 --lr 0.001 --defend_type SCL --weight_decay 1e-5 --rep_type 2

python main_defense_for_different_topology.py \
  --graph_type random \
  --gnn_checkpoint_path ./checkpoint/tool_attack/20250701_215131-hiddim_1024-heads_8-layers_2-epochs_50-lr_0.001-dropout_0.2-wd_0.0002.pth \
  --model_type gpt-4o-mini --topk 3

python main_defense_for_different_topology1.py  \
  --graph_type random \
  --gnn_checkpoint_path ./checkpoint_un2/tool_attack/20260607_225353-defend_type_SCL-hiddim_1024-heads_8-layers_2-epochs_50-lr_0.001-dropout_0.2-wd_1e-05_rep_type_2.pth \
  --model_type gpt-4o-mini --samples 60 --defend_type SCL --topk 3 --rep_type 2
```

---

## Computing ASR and AUROC

Evaluation JSON files are written to `result/`. Use `evaluate_output.py`:

| Module | ASR function | `answer_type` |
|--------|--------------|---------------|
| PI (MMLU / CSQA) | `cal_wrong` | `"choice"` |
| PI (GSM8K) | `cal_wrong` | `"number"` |
| MA / MA_CSQA | `cal_wrong_acc` | — |
| All (detection) | `cal_mean_AUROC` | — |

```python
import json
from evaluate_output import cal_wrong, cal_mean_AUROC

with open("./result/mmlu/random/your_result.json") as f:
    data = json.load(f)

print("ASR per turn:", cal_wrong(data, answer_type="choice"))
print("Mean AUROC:", cal_mean_AUROC(data))
```
