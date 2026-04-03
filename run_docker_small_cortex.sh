#!/bin/bash
# run_docker_small_cortex.sh
# 用于训练 Small Cortex (1.5B Base + 300M Cortex) 的启动脚本
# 支持通过环境变量覆盖模型路径

# 定义日志文件名（区分开来）
LOG_FILE="./log/small_cortex_$(date +%Y%m%d_%H%M%S).log"

# 清理旧容器和日志链接
docker rm -f cortex-small-cortex-job 2>/dev/null || true
rm output_small_cortex.log 2>/dev/null || true
ln -s $LOG_FILE output_small_cortex.log

echo "Starting Small Cortex training log at: $LOG_FILE"
echo "Model path: ${CORTEX_MODEL_PATH:-'(using config default)'}"

# 保存代码版本到日志（便于复现）
echo "=== Code Version ===" > $LOG_FILE
git rev-parse HEAD >> $LOG_FILE 2>/dev/null || echo "git not available" >> $LOG_FILE
git diff --stat >> $LOG_FILE 2>/dev/null || echo "no git diff" >> $LOG_FILE
echo "" >> $LOG_FILE
cat src/model_deep.py src/train_deep.py src/config.py >> $LOG_FILE

# 启动训练容器
docker run --gpus all -d \
    --net=host \
    --name cortex-small-cortex-job \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e PYTHONUNBUFFERED=1 \
    -e WANDB_CACHE_DIR="/home/alan/code/cortex-llm/wandb_cache" \
    -e CORTEX_MODEL_PATH="${CORTEX_MODEL_PATH:-}" \
    -v /home/alan/code:/home/alan/code \
    -v /home/alan/.cache/huggingface:/root/.cache/huggingface \
    -v /home/alan/.bashrc:/root/.bashrc \
    -w $(pwd) \
    cortex:v2 \
    bash -c "python -u src/train_deep.py >> $LOG_FILE 2>&1"

echo "Container started: cortex-small-cortex-job"
echo "To view logs: tail -f $LOG_FILE"
echo "To stop: docker stop cortex-small-cortex-job"
