"""Tests for ChatML formatting, checked against the model's real Jinja template.

Rather than assert on hand-written expected strings, these render the actual
``chat_template`` from ``tokenizer_config.json`` with Jinja and require an exact
match. That makes the test an oracle rather than a restatement of the
implementation: if the template and this module ever disagree, the test fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoinfer.chat import (
    DEFAULT_SYSTEM_PROMPT,
    ChatTemplate,
    Message,
    UnsupportedConversationError,
    encode_chat,
)
from nanoinfer.tokenizer import Tokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
CONFIG = MODEL_DIR / "tokenizer_config.json"

pytestmark = pytest.mark.skipif(not CONFIG.exists(), reason="model not downloaded")


@pytest.fixture(scope="module")
def jinja_render():
    """Render the model's own Jinja template, as transformers would."""
    pytest.importorskip("jinja2")
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    source = json.loads(CONFIG.read_text(encoding="utf-8"))["chat_template"]
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    template = env.from_string(source)

    def render(messages, add_generation_prompt=True):
        return template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tools=None,
        )

    return render


@pytest.fixture(scope="module")
def chat() -> ChatTemplate:
    return ChatTemplate.from_model_dir(MODEL_DIR)


CONVERSATIONS = [
    [{"role": "user", "content": "Hello"}],
    [{"role": "user", "content": "What is 2 + 2?"}],
    [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi"},
    ],
    [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Explain RoPE."},
    ],
    [
        {"role": "system", "content": "Answer in French."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Bonjour !"},
        {"role": "user", "content": "Merci"},
    ],
    [{"role": "user", "content": "multi\nline\ncontent"}],
    [{"role": "user", "content": "unicode 日本語 🦀 café"}],
    [{"role": "user", "content": ""}],
    [{"role": "user", "content": "trailing whitespace   "}],
    [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
        {"role": "user", "content": "e"},
    ],
]


@pytest.mark.reference
@pytest.mark.parametrize("messages", CONVERSATIONS)
@pytest.mark.parametrize("add_generation_prompt", [True, False])
def test_matches_the_models_jinja_template(
    chat, jinja_render, messages, add_generation_prompt
):
    assert chat.render(messages, add_generation_prompt) == jinja_render(
        messages, add_generation_prompt
    )


# -- behaviour -------------------------------------------------------------


def test_default_system_prompt_is_inserted(chat):
    out = chat.render([{"role": "user", "content": "Hi"}])
    assert DEFAULT_SYSTEM_PROMPT in out
    assert out.startswith("<|im_start|>system\n")


def test_supplied_system_prompt_replaces_the_default(chat):
    out = chat.render(
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi"},
        ]
    )
    assert "You are terse." in out
    assert DEFAULT_SYSTEM_PROMPT not in out


def test_leading_system_message_is_not_emitted_twice(chat):
    out = chat.render(
        [
            {"role": "system", "content": "UNIQUE_MARKER"},
            {"role": "user", "content": "Hi"},
        ]
    )
    assert out.count("UNIQUE_MARKER") == 1


def test_generation_prompt_opens_an_assistant_turn(chat):
    out = chat.render([{"role": "user", "content": "Hi"}])
    assert out.endswith("<|im_start|>assistant\n")
    assert not out.endswith("<|im_end|>\n")


def test_without_generation_prompt_the_last_turn_is_closed(chat):
    out = chat.render([{"role": "user", "content": "Hi"}], add_generation_prompt=False)
    assert out.endswith("<|im_end|>\n")


def test_message_objects_and_dicts_are_equivalent(chat):
    as_dicts = chat.render([{"role": "user", "content": "Hi"}])
    as_objects = chat.render([Message("user", "Hi")])
    assert as_dicts == as_objects


def test_rejects_unknown_role():
    with pytest.raises(UnsupportedConversationError, match="not implemented"):
        Message("wizard", "abracadabra")


def test_rejects_tool_role():
    """Refused rather than guessed: the tool path needs the Jinja template."""
    with pytest.raises(UnsupportedConversationError):
        Message("tool", "{}")


def test_rejects_tool_calls(chat):
    with pytest.raises(UnsupportedConversationError, match="tool_calls"):
        chat.render([{"role": "assistant", "content": "", "tool_calls": [{}]}])


def test_from_model_dir_rejects_a_different_default_prompt(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["chat_template"] = "{{ 'something else entirely' }}"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(UnsupportedConversationError, match="default system"):
        ChatTemplate.from_model_dir(tmp_path)


def test_from_model_dir_rejects_missing_template(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(UnsupportedConversationError, match="no chat_template"):
        ChatTemplate.from_model_dir(tmp_path)


# -- encoding --------------------------------------------------------------


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_model_dir(MODEL_DIR)


def test_encode_chat_emits_control_tokens_as_single_ids(tok, chat):
    ids = encode_chat(tok, [{"role": "user", "content": "Hi"}], template=chat)
    im_start = tok.token_to_id("<|im_start|>")
    im_end = tok.token_to_id("<|im_end|>")
    assert ids.count(im_start) == 3      # system, user, assistant
    assert ids.count(im_end) == 2        # system and user turns are closed
    # The generation prompt is <|im_start|> then "assistant" then a newline:
    # three tokens, the trailing newline included.
    assert ids[-3:] == [
        im_start,
        tok.token_to_id("assistant"),
        tok.token_to_id("Ċ"),
    ]


def test_encode_chat_round_trips(tok, chat):
    messages = [{"role": "user", "content": "Explain attention."}]
    rendered = chat.render(messages)
    assert tok.decode(encode_chat(tok, messages, template=chat)) == rendered


def test_generation_prompt_ends_mid_turn(tok, chat):
    """The last token must not be <|im_end|>, or the model has nothing to say."""
    ids = encode_chat(tok, [{"role": "user", "content": "Hi"}], template=chat)
    assert ids[-1] != tok.token_to_id("<|im_end|>")
