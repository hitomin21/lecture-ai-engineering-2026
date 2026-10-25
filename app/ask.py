"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --chat     # 会話履歴を保って続けて質問する

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from time import perf_counter

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def create_chat():
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)
    return client, chat


def ask(question: str, chat=None) -> str:
    if chat is None:
        client, chat = create_chat()

    started = perf_counter()
    response = chat.send_message(question)
    elapsed = perf_counter() - started

    usage = getattr(response, "usage_metadata", None) or {}
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_token_count")
        output_tokens = usage.get("candidates_token_count")
        total_tokens = usage.get("total_token_count")
    else:
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        output_tokens = getattr(usage, "candidates_token_count", None)
        total_tokens = getattr(usage, "total_token_count", None)

    print(f"[計測] 応答時間: {elapsed:.2f} 秒")
    if prompt_tokens is not None or output_tokens is not None or total_tokens is not None:
        print(f"[usage] 入力トークン数: {prompt_tokens if prompt_tokens is not None else 0}")
        print(f"[usage] 出力トークン数: {output_tokens if output_tokens is not None else 0}")
        print(f"[usage] 合計トークン数: {total_tokens if total_tokens is not None else 0}")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            client, chat = create_chat()
            question = args.question
            while True:
                if question is None:
                    try:
                        question = input("\n質問（終了するには /exit）: ").strip()
                    except EOFError:
                        break
                if question.lower() in ("/exit", "/quit"):
                    break
                if question:
                    print(f"\n[質問] {question}\n")
                    print(ask(question, chat))
                question = None
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
