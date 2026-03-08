#!/bin/bash
#
# CHTC Interactive Job에서 실행하여 Python 환경을 tarball로 만드는 스크립트
#
# 사용법:
#   1. 인터랙티브 잡 시작:
#      condor_submit -i chtc/build_env.sub
#   2. 인터랙티브 세션에서:
#      bash build_env.sh
#   3. 생성된 env.tar.gz를 /staging/$USER/ 로 복사
#
set -euo pipefail

echo "===== Building Safe-RLHF Python Environment ====="
echo "Date: $(date)"
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"

# ─── Use conda env that comes with the Docker image ──────────────────────────
# pytorch/pytorch image already has Python + PyTorch + CUDA
# We just install the remaining dependencies into a portable env

# Create a clean virtual env based on system Python
python3 -m venv --system-site-packages env
source env/bin/activate

# ─── Install remaining project dependencies ──────────────────────────────────
# Upgrade PyTorch to satisfy newest Transformers requirements.
pip install --no-warn-script-location \
    --index-url https://download.pytorch.org/whl/cu124 \
    'torch==2.4.1+cu124' \
    'torchvision==0.19.1+cu124'
pip install --no-warn-script-location \
    'transformers>=5.0.0' 'deepspeed>=0.12' 'accelerate>=0.25' \
    'datasets' 'tokenizers' \
    numpy scipy sentencepiece wandb tensorboard optree matplotlib tqdm rich

# NOTE: safe_rlhf 자체는 pip install 하지 않습니다.
# 학습 스크립트에서 PYTHONPATH로 프로젝트 루트를 잡기 때문에 패키지 설치 불필요.

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import safe_rlhf; print('safe-rlhf imported successfully')"

# Relocatable env: use relative paths
deactivate

echo "Packing environment into env.tar.gz ..."
tar -czf env.tar.gz env/

echo "===== env.tar.gz created ($(du -h env.tar.gz | cut -f1)) ====="
echo "Copy it to staging:"
echo "  cp env.tar.gz /staging/$USER/"
