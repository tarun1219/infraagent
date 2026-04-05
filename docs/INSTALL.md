# Installation Guide

## 1. Python Environment

```bash
python --version  # requires 3.10+
git clone https://github.com/tarun1219/infraagent.git
cd infraagent
pip install -r requirements.txt
```

## 2. LLM Inference (Ollama)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models used in the paper
ollama pull deepseek-coder-v2:16b-lite-instruct-q4_K_M   # ~9 GB — primary model
ollama pull codellama:13b-instruct-q4_K_M                 # ~7 GB
ollama pull mistral:7b-instruct-q4_K_M                   # ~4 GB
ollama pull phi3:3.8b-instruct-q4_K_M                    # ~2 GB
ollama pull llama3.1:70b-instruct-q4_K_M                 # ~40 GB — optional
ollama pull qwen2.5-coder:32b-instruct-q4_K_M            # ~18 GB — optional
```

For commercial models, set environment variables:
```bash
export OPENAI_API_KEY="sk-..."      # for GPT-4o (~$5 for 300 tasks)
export ANTHROPIC_API_KEY="sk-ant-..." # for Claude-3.5-Sonnet
```

## 3. IaC Validation Tools

### macOS
```bash
brew install hadolint kubeconform trivy kind kubectl
pip install checkov yamllint
```

### Ubuntu/Debian
```bash
# yamllint + checkov via pip
pip install yamllint checkov

# hadolint
wget -O /usr/local/bin/hadolint https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64
chmod +x /usr/local/bin/hadolint

# kubeconform
wget -O /tmp/kc.tar.gz https://github.com/yannh/kubeconform/releases/download/v0.6.4/kubeconform-linux-amd64.tar.gz
tar -xzf /tmp/kc.tar.gz -C /usr/local/bin/

# trivy
apt-get install -y wget apt-transport-https gnupg
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add -
apt-get install trivy

# Terraform
wget -O /tmp/tf.zip https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip /tmp/tf.zip -d /usr/local/bin/
```

## 4. Optional: Kubernetes Dry-Run Validation

```bash
bash scripts/setup_kind_cluster.sh
# Creates a kind cluster named "infraagent"
# Required for Layer 2.5 server-side validation
```

## 5. Optional: Terraform LocalStack Testing

```bash
bash scripts/setup_localstack.sh
# Starts LocalStack on port 4566
# Required for runtime deployment gap analysis
```

## 6. Verify Installation

```bash
make test
# All unit tests should pass
# Some tests skip gracefully if external tools are missing
```
