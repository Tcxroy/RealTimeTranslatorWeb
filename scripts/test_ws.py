"""
SyncVoice 快速集成测试
测试项: WebSocket 连接 → 发送模拟音频 → 接收 ASR/翻译流
用法: python3 scripts/test_ws.py
"""
import asyncio
import base64
import json
import numpy as np
import websockets

WS_URL = "ws://localhost:8765/ws/interpret"

async def test():
    print(f"\n🔌 连接 {WS_URL}…")
    try:
        async with websockets.connect(WS_URL) as ws:
            print("✓ 连接成功")

            # 1. 接收 ready 消息
            msg = json.loads(await ws.recv())
            print(f"  服务器: {msg}")

            # 2. 发送配置
            await ws.send(json.dumps({"type":"config","src_lang":"zh","tgt_lang":"English","model":"qwen3.5:9b"}))
            msg = json.loads(await ws.recv())
            print(f"  配置确认: {msg}")

            # 3. 发送静音测试 (1秒 16kHz float32)
            print("\n📤 发送 1s 静音测试…")
            silence = np.zeros(16000, dtype=np.float32)
            b64 = base64.b64encode(silence.tobytes()).decode()
            await ws.send(b64)

            # 4. 读取几条 VAD 消息
            for _ in range(3):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    msg = json.loads(raw)
                    print(f"  VAD: voice={msg.get('voice')}, conf={msg.get('conf')}")
                except asyncio.TimeoutError:
                    break

            # 5. Ping
            await ws.send(json.dumps({"type":"ping"}))
            pong = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
            print(f"\n  Ping-Pong: {pong['type']} ✓")

            print("\n✅ 基础测试通过！服务器工作正常。\n")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        print("  请先运行: bash scripts/run.sh\n")

if __name__ == "__main__":
    asyncio.run(test())
