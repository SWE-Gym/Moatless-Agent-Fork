#!/bin/bash

eval "$(conda shell.bash hook)"

export OH_PATH=<PATH_TO_OPENHANDS_REPO>
export ML_PATH=<PATH_TO_MOATLESS_REPO>

export PRED_PATH="/home/jiayipan/code/24FA/temp/ml-01/moatless-tools/t/zero-test-20241122_moatless_openai_qwen_temp_1.0"

python ${ML_PATH}/scripts/merge_jsonl.py --exp_path ${PRED_PATH}
# bash eval_all.sh
conda activate oh
export ALLHANDS_API_KEY=<OPEN_HANDS_CLOUD_API>
cd ${OH_PATH}
evaluation/swe_bench/scripts/cleanup_remote_runtime.sh
RUNTIME=remote SANDBOX_REMOTE_RUNTIME_API_URL="https://runtime.eval.all-hands.dev" EVAL_DOCKER_IMAGE_PREFIX="us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images" \
evaluation/swe_bench/scripts/eval_infer_remote.sh \
${PRED_PATH}/preds.jsonl \
128 "princeton-nlp/SWE-bench_Lite" "test"