#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  SyncVoice — MacBook Pro M4 一键启动脚本
#  用法: bash scripts/run.sh
# ─────────────────────────────────────────────────────────────
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_DIR="$PROJECT_ROOT/server"
VENV_DIR="$PROJECT_ROOT/.venv"
PORT=8765

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║   SyncVoice — 实时同传翻译服务器     ║"
echo "  ║   MacBook Pro M4 · 36GB Unified RAM  ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# ── 1. 检查 Ollama ──────────────────────────────────────────
echo "▶ 检查 Ollama 状态…"
if ! command -v ollama &>/dev/null; then
  echo "  ✗ 未找到 ollama，请先安装: https://ollama.ai"
  exit 1
fi

if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
  echo "  ○ 启动 Ollama 守护进程…"
  ollama serve &>/dev/null &
  sleep 3
fi

echo "  ✓ Ollama 运行中"

# 检查模型是否已下载
for MODEL in "qwen3.5:9b" "translategemma:12b"; do
  if ollama list 2>/dev/null | grep -q "${MODEL%%:*}"; then
    echo "  ✓ 模型已就绪: $MODEL"
  else
    echo "  ○ 拉取模型 $MODEL（首次运行，请等待）…"
    ollama pull "$MODEL"
  fi
done

# ── 2. Python 虚拟环境 ──────────────────────────────────────
echo ""
echo "▶ 检查 Python 环境…"
if ! command -v python3 &>/dev/null; then
  echo "  ✗ 未找到 python3，请安装 Python 3.11+"
  exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  ✓ Python $PY_VER"

if [ ! -d "$VENV_DIR" ]; then
  echo "  ○ 创建虚拟环境…"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "  ○ 安装/更新依赖（首次约需 2-3 分钟）…"
pip install -q --upgrade pip
pip install -q -r "$SERVER_DIR/requirements.txt"
echo "  ✓ 依赖就绪"

# ── 3. 检查端口 ──────────────────────────────────────────────
echo ""
echo "▶ 检查端口 $PORT…"
if lsof -ti:$PORT &>/dev/null; then
  echo "  ⚠ 端口 $PORT 已被占用，尝试终止旧进程…"
  lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
  sleep 1
fi
echo "  ✓ 端口 $PORT 可用"

# ── 4. 启动 FastAPI 服务器 ───────────────────────────────────
echo ""
echo "▶ 启动 SyncVoice 服务器…"
echo "  WebSocket: ws://localhost:$PORT/ws/interpret"
echo "  Health:    http://localhost:$PORT/health"
echo "  前端页面:  file://$PROJECT_ROOT/index.html"
echo ""
echo "  按 Ctrl+C 停止服务"
echo "─────────────────────────────────────────────────────────"

cd "$SERVER_DIR"
exec uvicorn main:app \
  --host 0.0.0.0 \
  --port $PORT \
  --workers 1 \
  --loop asyncio \
  --log-level info
