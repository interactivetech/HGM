# 07/07/2026: run miniswebench HGM
export VLLM_BASE_URL="http://173.73.39.103:8000/v1"
export VLLM_API_KEY="dummy"
export VLLM_MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

export LLM="vllm"
export HOURS_PER_TASK=1
export NUM_WORKERS=1

./run_test.sh