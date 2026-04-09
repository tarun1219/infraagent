# Installation Guide

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11 |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB | 120 GB (all 6 models) |
| OS | macOS 13+, Ubuntu 22.04 | Same |
| Ollama | 0.1.27+ | Latest |
| Docker | 24+ (for LocalStack / kind) | Latest |

GPU is **not required** — all models run via Ollama CPU inference with 4-bit quantization.
A 16 GB RAM machine (e.g., M2 Pro MacBook) is sufficient for models up to 16B parameters.

---

## 1. Clone the Repository

```bash
git clone https://github.com/tarun1219/infraagent.git
cd infraagent
```

---

## 2. Python Environment

### Option A — conda (recommended)

```bash
conda create -n infraagent python=3.11 -y
conda activate infraagent
pip install -r requirements.txt
pip install -e .
```

### Option B — venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Verify:

```bash
python -c "import iachench; print('iachench OK')"
```

---

## 3. Ollama — Local LLM Inference

Ollama runs open-source models locally with 4-bit quantization (Q4_K_M GGUF format),
reducing memory footprint by ~4× with negligible accuracy loss.

### Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS via Homebrew
brew install ollama

# Verify (requires 0.1.27+)
ollama --version
```

Start the Ollama daemon (runs in background):

```bash
ollama serve &
```

### Pull Models

Download only the models you need. DeepSeek Coder v2 is the primary model from the paper.

```bash
# Primary model — best open-source result in the paper
ollama pull deepseek-coder-v2:16b-lite-instruct-q4_K_M   # ~9.1 GB

# Additional models evaluated in the paper
ollama pull codellama:13b-instruct-q4_K_M                # ~7.4 GB
ollama pull mistral:7b-instruct-q4_K_M                   # ~4.1 GB
ollama pull phi3:3.8b-instruct-q4_K_M                    # ~2.3 GB

# Large models (optional — need 40+ GB RAM)
ollama pull llama3.1:70b-instruct-q4_K_M                 # ~40 GB
ollama pull qwen2.5-coder:32b-instruct-q4_K_M            # ~18 GB
```

Verify a model is working:

```bash
ollama run deepseek-coder-v2:16b-lite-instruct-q4_K_M \
  "Write a minimal Kubernetes Deployment YAML for nginx."
```

### Commercial API Models (Optional)

```bash
export OPENAI_API_KEY="sk-..."           # GPT-4o (~$5 per 300-task run)
export ANTHROPIC_API_KEY="sk-ant-..."    # Claude 3.5 Sonnet (~$6 per 300-task run)
```

---

## 4. ChromaDB — RAG Vector Store

ChromaDB is installed automatically via `requirements.txt`. The embedding model
(`all-MiniLM-L6-v2`, ~90 MB) is downloaded on first run.

Verify ChromaDB and embeddings are working:

```bash
python -c "
import chromadb
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('ChromaDB OK — embedding dim:', model.get_sentence_embedding_dimension())
"
```

The RAG corpus (Kubernetes, Terraform, Dockerfile documentation) lives in `rag_corpus/`
and is indexed automatically on the first evaluation run. To pre-build the index:

```bash
python -c "
from infraagent.agent import InfraAgent
InfraAgent(use_rag=True).build_rag_index()
print('RAG index built.')
"
```

---

## 5. IaC Validation Tools

### macOS

```bash
brew install hadolint kubeconform trivy kind kubectl terraform
pip install checkov yamllint
```

### Ubuntu 22.04

```bash
# Python tools
pip install checkov yamllint

# hadolint
sudo wget -O /usr/local/bin/hadolint \
  https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64
sudo chmod +x /usr/local/bin/hadolint

# kubeconform
wget -O /tmp/kc.tar.gz \
  https://github.com/yannh/kubeconform/releases/download/v0.6.4/kubeconform-linux-amd64.tar.gz
sudo tar -xzf /tmp/kc.tar.gz -C /usr/local/bin/

# trivy
sudo apt-get install -y wget apt-transport-https gnupg lsb-release
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" \
  | sudo tee /etc/apt/sources.list.d/trivy.list
sudo apt-get update && sudo apt-get install -y trivy

# Terraform
wget -O /tmp/tf.zip \
  https://releases.hashicorp.com/terraform/1.7.0/terraform_1.7.0_linux_amd64.zip
sudo unzip -o /tmp/tf.zip -d /usr/local/bin/
```

---

## 6. Kubernetes Server-Side Dry-Run (Layer 2.5)

Required for `kubectl dry-run=server` validation. The pipeline runs without this —
Layer 2.5 is skipped gracefully when no cluster is available.

```bash
# Install kind
brew install kind    # macOS
# Linux: https://kind.sigs.k8s.io/docs/user/quick-start/#installation

# Create the evaluation cluster (~2 min)
bash scripts/setup_kind_cluster.sh

# Verify
kubectl cluster-info --context kind-infraagent-eval
```

---

## 7. LocalStack — Terraform Deployment Testing

Required only for the deployment gap analysis (Figure 12). Skipped gracefully otherwise.

```bash
# Docker must be running
bash scripts/setup_localstack.sh

# Verify
curl -s http://localhost:4566/_localstack/health | python3 -m json.tool
```

---

## 8. Verify Full Installation

```bash
make test
```

Expected output:
```
tests/test_validators.py ......    [PASSED]
tests/test_benchmark_loading.py .. [PASSED]
```

Tests that require Ollama, kind, or LocalStack are skipped gracefully when those
services are not running.

---

## Troubleshooting

**`ollama: command not found` after install**
```bash
export PATH="$HOME/.ollama/bin:$PATH"
# or restart your terminal
```

**`Error: model not found`**
```bash
ollama list           # see what's already pulled
ollama pull <model>   # pull the missing model
```

**`chromadb.errors.InvalidDimensionException`**
```bash
rm -rf .chroma/    # delete stale index; it will rebuild on next run
```

**`checkov` is slow on first run**
Checkov downloads policy bundles (~60 s) on the first invocation. Subsequent runs are fast.

**`kubectl dry-run` returns `connection refused`**
The kind cluster isn't running. Recreate it:
```bash
kind delete cluster --name infraagent-eval
bash scripts/setup_kind_cluster.sh
```

**`terraform init` fails inside validator**
The validator runs `terraform init` in a temp directory and needs internet access to
download providers. Cache them to avoid repeated downloads:
```bash
export TF_PLUGIN_CACHE_DIR="$HOME/.terraform.d/plugin-cache"
mkdir -p "$TF_PLUGIN_CACHE_DIR"
```

**Out of memory with 70B model**
Use a smaller quantization:
```bash
ollama pull llama3.1:70b-instruct-q2_K   # ~25 GB instead of ~40 GB
```
