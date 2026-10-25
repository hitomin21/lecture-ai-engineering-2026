"""app.ask のテスト。"""

import weakref

from app.ask import ask


def test_ask_prints_usage_metadata(monkeypatch, capsys):
    class FakeUsageMetadata:
        prompt_token_count = 123
        candidates_token_count = 45
        total_token_count = 168

    class FakeResponse:
        text = "これはテストです。"
        usage_metadata = FakeUsageMetadata()

    class FakeChat:
        def send_message(self, contents):
            assert contents == "テスト質問"
            return FakeResponse()

    class FakeChats:
        def create(self, model):
            assert model == "test-model"
            return FakeChat()

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.chats = FakeChats()

    monkeypatch.setattr("app.ask.MODEL", "test-model")
    monkeypatch.setattr("app.ask.get_api_key", lambda: "test-key")
    monkeypatch.setattr("app.ask.genai.Client", FakeClient)
    monkeypatch.setattr("app.ask.perf_counter", iter([1.0, 2.25]).__next__)

    assert ask("テスト質問") == "これはテストです。"

    captured = capsys.readouterr().out
    assert "応答時間: 1.25 秒" in captured
    assert "入力トークン数: 123" in captured
    assert "出力トークン数: 45" in captured
    assert "合計トークン数: 168" in captured


def test_ask_reuses_chat_for_conversation(capsys):
    from app.ask import ask

    class FakeResponse:
        text = "回答"
        usage_metadata = None

    class FakeChat:
        messages = []

        def send_message(self, contents):
            self.messages.append(contents)
            return FakeResponse()

    chat = FakeChat()
    assert ask("最初の質問", chat) == "回答"
    assert ask("続きの質問", chat) == "回答"
    assert chat.messages == ["最初の質問", "続きの質問"]


def test_ask_keeps_client_alive_while_sending(monkeypatch, capsys):
    client_reference = None

    class FakeResponse:
        text = "回答"
        usage_metadata = None

    class FakeChat:
        def send_message(self, contents):
            assert client_reference() is not None
            return FakeResponse()

    class FakeChats:
        def create(self, model):
            return FakeChat()

    class FakeClient:
        def __init__(self, api_key):
            nonlocal client_reference
            client_reference = weakref.ref(self)
            self.chats = FakeChats()

    monkeypatch.setattr("app.ask.get_api_key", lambda: "test-key")
    monkeypatch.setattr("app.ask.genai.Client", FakeClient)

    assert ask("テスト質問") == "回答"
