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
    monkeypatch.setattr(ask, "compact_chat", lambda client, chat, summary, question: (chat, summary))

    def interrupted_input(_):
        raise end

    monkeypatch.setattr("builtins.input", interrupted_input)
    ask.chat_loop("質問")
    client.__exit__.assert_called_once()


def make_history(turns):
    return [
        types.Content(role=role, parts=[types.Part(text=f"{turn}:{role}")])
        for turn in range(turns)
        for role in ("user", "model")
    ]


def summary_response(text, finish_reason="STOP"):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        finish_reason=finish_reason,
    )])


@pytest.mark.parametrize("turns,tokens", [(3, 10000), (4, 4000)])
def test_compact_chat_keeps_history_when_not_needed(turns, tokens):
    client = Mock()
    client.models.count_tokens.return_value = types.CountTokensResponse(total_tokens=tokens)
    chat = Chat(modules=Mock(), model=ask.MODEL, config=None, history=make_history(turns))

    result, summary = ask.compact_chat(client, chat, "", "次の質問")

    assert result is chat
    assert summary == ""
    client.models.generate_content.assert_not_called()
    client.chats.create.assert_not_called()


def test_compact_chat_replaces_old_history_and_merges_previous_summary():
    client = Mock()
    client.models.count_tokens.return_value = types.CountTokensResponse(total_tokens=5000)
    client.models.generate_content.side_effect = [
        summary_response("最初の要約"), summary_response("統合した要約")
    ]
    modules = Mock()
    client.chats.create.side_effect = lambda **kwargs: Chat(modules=modules, **kwargs)
    history = make_history(5)
    chat = Chat(modules=modules, model=ask.MODEL, config=None, history=history)

    new_chat, summary = ask.compact_chat(client, chat, "", "今回の質問")

    assert new_chat.get_history() == history[-6:]
    assert chat.get_history() == history  # 元の履歴は成功するまで変更しない
    assert summary == "最初の要約"
    first_prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "0:user" in first_prompt and "1:model" in first_prompt
    assert "2:user" not in first_prompt  # 直近3往復を要約に巻き込まない
    count_contents = client.models.count_tokens.call_args.kwargs["contents"]
    assert count_contents[-1].parts[0].text == "今回の質問"
    config = client.chats.create.call_args.kwargs["config"]
    assert "最初の要約" in config.system_instruction

    # 次の往復の後で再圧縮し、前回の要約も統合する。
    next_history = [*new_chat.get_history(), *make_history(1)]
    continued_chat = Chat(modules=modules, model=ask.MODEL, config=config, history=next_history)
    final_chat, summary = ask.compact_chat(client, continued_chat, summary, "さらに質問")

    assert summary == "統合した要約"
    assert final_chat.get_history() == next_history[-6:]
    assert "最初の要約" in client.models.generate_content.call_args.kwargs["contents"]
    counted_summary = client.models.count_tokens.call_args.kwargs["contents"][0]
    assert "最初の要約" in counted_summary.parts[0].text
    assert "統合した要約" in client.chats.create.call_args.kwargs["config"].system_instruction


@pytest.mark.parametrize("text,reason", [("", "STOP"), ("途中の要約", "MAX_TOKENS")])
def test_compact_chat_preserves_history_on_failed_summary(text, reason):
    client = Mock()
    client.models.count_tokens.return_value = types.CountTokensResponse(total_tokens=5000)
    client.models.generate_content.return_value = summary_response(text, reason)
    chat = Chat(modules=Mock(), model=ask.MODEL, config=None, history=make_history(4))

    result, summary = ask.compact_chat(client, chat, "前回の要約", "質問")

    assert result is chat
    assert summary == "前回の要約"
    client.chats.create.assert_not_called()
