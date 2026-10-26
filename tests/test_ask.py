"""実際のAPIを呼ばず、SDKが次の質問に会話履歴を含めることを確認する。"""

from unittest.mock import Mock

import pytest
from google.genai import types
from google.genai.chats import Chat

from app import ask


def test_chat_loop_sends_previous_question_and_answer(monkeypatch, capsys):
    modules = Mock()
    modules.generate_content.return_value = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model", parts=[types.Part(text="LoRAは学習手法です。")]
                ),
                finish_reason="STOP",
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5
        ),
    )
    chat = Chat(modules=modules, model=ask.MODEL, config=None, history=[])
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.chats.create.return_value = chat
    monkeypatch.setattr(ask.genai, "Client", Mock(return_value=client))
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    inputs = iter(["  ", "その利点は？", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    ask.chat_loop("LoRAとは？")

    assert modules.generate_content.call_count == 2
    contents = modules.generate_content.call_args.kwargs["contents"]
    assert [content.role for content in contents] == ["user", "model", "user"]
    assert [content.parts[0].text for content in contents] == [
        "LoRAとは？", "LoRAは学習手法です。", "その利点は？"
    ]
    client.chats.create.assert_called_once_with(model=ask.MODEL)
    output = capsys.readouterr().out
    assert "[入力トークン] 10" in output
    assert "[出力トークン] 5" in output


@pytest.mark.parametrize("end", [EOFError, KeyboardInterrupt])
def test_chat_loop_exits_on_terminal_interrupt(monkeypatch, end):
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(ask.genai, "Client", Mock(return_value=client))
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask, "ask", Mock(return_value="回答"))

    def interrupted_input(_):
        raise end

    monkeypatch.setattr("builtins.input", interrupted_input)
    ask.chat_loop("質問")
    client.__exit__.assert_called_once()
