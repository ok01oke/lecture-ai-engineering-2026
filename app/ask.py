"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat "LoRAとは何ですか？"  # 履歴を保って続けて質問する

ツールを持たない「素の」呼び出し。--chat では実行中だけ会話履歴を保持する。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors
from google.genai.chats import Chat

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, chat: Chat | None = None) -> str:
    """同じ chat を渡すと、それまでの質問と回答を含めて問い合わせる。"""
    if chat is None:
        with genai.Client(api_key=get_api_key()) as client:
            return ask(question, client.chats.create(model=MODEL))

    response = chat.send_message(question)

    usage = response.usage_metadata
    if usage is not None:
        print(f"[入力トークン] {usage.prompt_token_count}")
        print(f"[出力トークン] {usage.candidates_token_count}")

    return response.text or "（テキストの回答がありませんでした）"


def chat_loop(question: str) -> None:
    """1つのチャットを使い回し、終了まで質問を受け付ける。"""
    print("続けて質問できます。終了するには /exit または /quit を入力してください。")
    with genai.Client(api_key=get_api_key()) as client:
        chat = client.chats.create(model=MODEL)
        while True:
            print(ask(question, chat))
            while True:
                try:
                    question = input("\n[次の質問] ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                if question.lower() in {"/exit", "/quit"}:
                    return
                if question:
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        if args.chat:
            chat_loop(args.question)
        else:
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
