# Safe-RLHF on CHTC (HTCondor) 실행 가이드

## 📋 개요

이 가이드는 Safe-RLHF 프로젝트를 UW-Madison CHTC 환경에서 HTCondor GPU 잡으로
실행하는 방법을 설명합니다.

Safe-RLHF 파이프라인은 총 **4단계**로 구성됩니다:

```
1. SFT (Supervised Fine-Tuning)   →  기본 모델 미세조정
2. Reward Model 학습              →  보상 모델 학습
3. Cost Model 학습                →  비용(안전) 모델 학습
4. PPO-Lagrangian RLHF            →  강화학습 기반 정렬
```

---

## 🔧 사전 준비

### 1. 로그 디렉토리 생성

```bash
mkdir -p outputs/logs
```

### 2. 실행 스크립트 권한 설정

```bash
chmod +x chtc/run_sft.sh chtc/run_reward.sh chtc/run_ppo.sh chtc/build_env.sh
```

### 3. (권장) Python 환경을 미리 빌드

매 잡마다 pip install을 하면 시간이 오래 걸리므로, **환경 tarball**을 미리 만들어
`/staging` 에 올려두는 것을 권장합니다.

```bash
# 인터랙티브 잡 시작
condor_submit -i chtc/build_env.sub

# 인터랙티브 세션 안에서:
bash build_env.sh

# 완료 후 staging으로 복사
cp env.tar.gz /staging/$USER/
exit
```

### 4. (대형 모델) 모델 가중치를 staging에 배치

HuggingFace에서 모델을 자동 다운로드하면 잡 실행 시간이 크게 늘어납니다.
미리 다운로드하여 `/staging` 에 올려두세요:

```bash
# 로컬(로그인 노드)에서 모델 다운로드
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'huggyllama/llama-7b'
AutoTokenizer.from_pretrained(model_name, cache_dir='./llama-7b')
AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./llama-7b')
"

# tarball로 압축 후 staging에 배치
tar -czf llama-7b.tar.gz llama-7b/
cp llama-7b.tar.gz /staging/$USER/models/
```

---

## 🚀 잡 제출 방법

### Step 1: SFT (Supervised Fine-Tuning)

```bash
condor_submit chtc/sft.sub
```

기본 설정:
- 모델: `huggyllama/llama-7b`
- GPU: A100 40GB+ × 1
- 메모리: 64GB
- 예상 소요 시간: 수 시간 ~ 1일

### Step 2: Reward Model

```bash
condor_submit chtc/reward.sub
```

### Step 3: Cost Model

```bash
condor_submit chtc/cost.sub
```

### Step 4: PPO-Lagrangian

```bash
condor_submit chtc/ppo.sub
```

---

## 📂 파일 구조

```
chtc/
├── README.md           ← 이 파일
├── build_env.sh        ← Python 환경 빌드 스크립트
├── build_env.sub       ← 환경 빌드용 인터랙티브 잡
├── sft.sub             ← SFT 학습 제출 파일
├── reward.sub          ← Reward Model 학습 제출 파일
├── cost.sub            ← Cost Model 학습 제출 파일
├── ppo.sub             ← PPO-Lag 학습 제출 파일
├── run_sft.sh          ← SFT 실행 래퍼 스크립트
├── run_reward.sh       ← Reward Model 실행 래퍼 스크립트
├── run_cost.sh         ← Cost Model 실행 래퍼 스크립트
├── run_ppo.sh          ← PPO-Lag 실행 래퍼 스크립트
└── logs/               ← HTCondor 로그 디렉토리
```

---

## ⚙️ 커스터마이징

### submit 파일에서 자주 변경하는 항목

| 항목 | 설명 | 예시 |
|------|------|------|
| `docker_image` | 사용할 Docker 이미지 | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` |
| `arguments` | 학습 인자 (모델 경로, 출력 경로 등) | `huggyllama/llama-7b output/sft 3 none` |
| `request_gpus` | GPU 수 | `1` 또는 `2` |
| `request_memory` | 메모리 | `64GB`, `128GB` |
| `+GPUJobLength` | 잡 길이 | `"short"`, `"medium"`, `"long"` |
| `require_gpus` | GPU 스펙 요구 | `(GlobalMemoryMb >= 40000)` |
| `transfer_input_files` | 전송할 파일 | staging 파일 경로 추가 |

### staging 사용 시 (`/staging/$USER`)

submit 파일에서 주석 처리된 부분을 해제하세요:

```
transfer_input_files = safe_rlhf/, ..., \
                       /staging/slyu59/env.tar.gz, \
                       /staging/slyu59/models/llama-7b.tar.gz
```

staging을 사용할 경우 requirements에 추가:
```
requirements = (Target.HasCHTCStaging == true)
```

### 작은 모델로 테스트

GPU 자원이 제한적이면 작은 모델로 먼저 테스트하세요:

```
arguments = gpt2 output/sft-test 3 none
```

그리고 리소스를 줄입니다:
```
request_gpus = 1
request_memory = 16GB
require_gpus = (GlobalMemoryMb >= 8000)
+GPUJobLength = "short"
```

---

## 🔍 잡 모니터링

```bash
# 큐에 있는 잡 확인
condor_q

# 잡 상세 정보
condor_q -better-analyze <JOB_ID>

# 실행 중 로그 확인
tail -f outputs/logs/sft/<CLUSTER>/0.out

# 잡 제거
condor_rm <JOB_ID>

# 잡 이력 확인
condor_history <JOB_ID>
```

---

## ⚠️ 주의사항

1. **CHTC 홈 디렉토리 용량**: 홈은 ~20GB 제한. 대용량 파일(모델, 출력)은
   반드시 `/staging/$USER/` 를 사용하세요.

2. **네트워크 접근**: 실행 노드에서 인터넷 접근이 제한될 수 있습니다.
   모델과 데이터셋을 미리 다운로드하여 staging에 올리세요.

3. **Preemption**: GPU 잡은 preemption될 수 있습니다.
   `+is_resumable = true` 와 `max_retries` 가 설정되어 있습니다.

4. **WANDB**: 실행 노드에서 인터넷이 안 될 수 있으므로 `WANDB_MODE=offline`
   으로 설정되어 있습니다. 잡 완료 후 `wandb sync` 로 동기화하세요.

5. **멀티 GPU**: `request_gpus = 2` 이상 설정 시 매칭 가능한 노드가 적어
   대기 시간이 길어질 수 있습니다.

6. **학습 순서**: 반드시 SFT → Reward/Cost Model → PPO 순서로 실행하세요.
   각 단계의 출력이 다음 단계의 입력이 됩니다.

---

## 🆘 문제 해결

| 증상 | 해결 방법 |
|------|-----------|
| 잡이 idle 상태로 오래 대기 | `condor_q -better-analyze` 로 원인 확인. GPU 스펙 요구를 낮추거나 `request_gpus` 줄이기 |
| Hold 상태 | `condor_q -hold` 로 이유 확인. Docker 이미지/파일 경로 문제일 수 있음 |
| OOM (메모리 부족) | `request_memory` 늘리기, `--offload optimizer` 사용, 배치 사이즈 줄이기 |
| GPU OOM | `--zero_stage 3 --offload all` 사용, `--per_device_train_batch_size` 줄이기 |
| 파일 전송 실패 | 경로 확인, staging 권한 확인 |
