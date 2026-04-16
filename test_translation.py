#!/usr/bin/env python3
"""测试翻译功能"""
import asyncio
import httpx
import json

OLLAMA_BASE_URL = "http://localhost:11434"

async def test_translation():
    """测试 Ollama 翻译"""

    system_prompt = (
        "You are a professional real-time simultaneous interpreter. "
        "Translate from Chinese to English.\n"
        "Rules:\n"
        "  1. Output ONLY the translation — no explanations, no notes.\n"
        "  2. Preserve proper nouns, technical terms, and speaker tone.\n"
        "  3. Be concise; this is live interpretation.\n"
    )

    payload = {
        "model": "qwen3.5:9b",
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": "你好，世界"},
        ],
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 256,
            "num_ctx": 1024,
        },
    }

    print("开始测试翻译...")
    print(f"发送请求到 {OLLAMA_BASE_URL}/api/chat")
    print(f"模型: {payload['model']}")
    print(f"输入: {payload['messages'][1]['content']}")

    translated_tokens = []

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            ) as resp:
                print(f"\n响应状态: {resp.status_code}")
                resp.raise_for_status()

                print("\n接收到的 token:")
                async for raw_line in resp.aiter_lines():
                    if not raw_line.strip():
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        translated_tokens.append(token)
                        print(f"  Token: '{token}'")

                    if chunk.get("done", False):
                        print("\n翻译完成")
                        break

    except httpx.HTTPError as exc:
        print(f"\nOllama HTTP 错误: {exc}")
        return
    except Exception as exc:
        print(f"\n意外错误: {exc}")
        import traceback
        traceback.print_exc()
        return

    translated = "".join(translated_tokens).strip()
    print(f"\n最终翻译结果: '{translated}'")
    print(f"Token 数量: {len(translated_tokens)}")

    if not translated:
        print("\n⚠️  警告: 翻译结果为空!")
        print("可能的原因:")
        print("  1. Ollama 模型没有正确响应")
        print("  2. 系统提示词格式不正确")
        print("  3. 模型配置问题")
    else:
        print("\n✓ 翻译功能正常!")

if __name__ == "__main__":
    asyncio.run(test_translation())
