import base64
import json
import requests
from pydantic import BaseModel, Field, ValidationError
from typing import List
import os
from json_repair import repair_json
# --- 配置 ---
# 建议使用 llama3.2-vision 或 minicpm-v (擅长中文OCR)
# 如果你确定本地有 qwen3-vl:8b 请保持，否则建议改为 "llama3.2-vision"
MODEL_NAME = "qwen3-vl:8b"
OLLAMA_API_URL = "http://10.0.0.173:11434/api/chat"  # 改用 chat 接口
TEST_IMAGE_PATH = "tutu.jpg"


# --- 数据模型 ---
class Ingredient(BaseModel):
    name: str = Field(..., description="食材名称")
    quantity: str = Field(..., description="数量")
    category: str = Field(..., description="类别")


class FridgeContent(BaseModel):
    items: List[Ingredient]


def encode_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def recognize_ingredients(image_path: str):
    print(f"🔍 读取图片: {image_path}")
    try:
        base64_image = encode_image(image_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # 提示词微调：对于视觉模型，指令越直接越好
    prompt = (
        "识别图片中的冰箱食材。输出 JSON 格式，包含 items 列表。"
        "每个 item 需有: name, quantity (带量词), category。"
        "不要输出任何 Markdown 标记或额外文本，只输出纯 JSON。"
    )

    # 构造 Chat 格式的 payload
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [base64_image]
        }],
        "stream": False,
        "format": "json",  # 强制 JSON
        "options": {
            "temperature": 0.1  # 低温更适合结构化输出
        }
    }

    print(f"🚀 发送请求给 Ollama ({MODEL_NAME})...")

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        # 兼容 generate 和 chat 接口的获取方式
        raw_content = result.get('response', '') or result.get('message', {}).get('content', '')

        print("-" * 30)
        print("🤖 模型原始输出 (已截取前100字符):", raw_content[:100].replace('\n', ' '))
        print("-" * 30)

        # === 核心修改点 ===
        # 使用 repair_json 自动修复坏掉的 JSON 字符串
        # 它会自动忽略前面的乱码，提取出有效的 JSON 对象
        cleaned_json_str = repair_json(raw_content, return_objects=False)

        print(f"🧹 清洗后的 JSON: {cleaned_json_str[:100]}...")

        # 解析清洗后的数据
        parsed_data = FridgeContent.model_validate_json(cleaned_json_str)
        # =================

        print("\n✅ Pydantic 验证成功!")
        for item in parsed_data.items:
            print(f"  - {item.name}: {item.quantity}")
        return parsed_data

    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        # 打印完整内容以便调试
        print(f"原始内容: {raw_content}")

    except requests.exceptions.RequestException as e:
        print(f"❌ 网络请求错误: {e}")
    except ValidationError as e:
        print(f"❌ JSON 校验失败: {e}")
        # 如果解析失败，打印原始内容方便调试
        print(f"导致错误的原始内容: {raw_content}")
    except Exception as e:
        print(f"❌ 未知错误: {e}")


if __name__ == "__main__":
    recognize_ingredients(TEST_IMAGE_PATH)