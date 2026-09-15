"""ChatML prompt formatting for the Instruct model.

An instruction-tuned model is not a text continuer. It was fine-tuned on a
specific control-token layout, and feeding it a bare prompt gets a bare
continuation rather than an answer. Qwen2.5-Instruct uses ChatML:

    <|im_start|>system
    ...system prompt...<|im_end|>
    <|im_start|>user
    ...question...<|im_end|>
    <|im_start|>assistant

The trailing ``<|im_start|>assistant\\n`` with no closing tag is the generation
prompt: it puts the model mid-turn so the next token it produces is the start
of the reply.

The model ships this layout as a Jinja template in ``tokenizer_config.json``.
Implementing a Jinja interpreter would be a project of its own, so this module
implements the conversational path directly -- system, user and assistant turns
-- and refuses anything else rather than guessing. The tests render the real
Jinja template with the real engine and require an exact match, so "implements
the same thing" is checked rather than asserted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# Qwen2.5's built-in system prompt, used when a conversation supplies none.
# It is baked into the template, so omitting it changes the prompt the model
# sees and therefore its behaviour.
DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

_CONVERSATIONAL_ROLES = frozenset({"system", "user", "assistant"})


class UnsupportedConversationError(ValueError):
    """The conversation uses a feature this formatter does not implement."""


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in _CONVERSATIONAL_ROLES:
            raise UnsupportedConversationError(
                f"role {self.role!r} is not implemented. This formatter covers "
                f"{sorted(_CONVERSATIONAL_ROLES)}; tool calls and tool responses "
                "need the Jinja template."
            )


def _coerce(messages: Iterable[Message | dict]) -> list[Message]:
    out: list[Message] = []
    for item in messages:
        if isinstance(item, Message):
            out.append(item)
            continue
        if item.get("tool_calls"):
            raise UnsupportedConversationError(
                "tool_calls are not implemented; use the Jinja template for those"
            )
        out.append(Message(role=item["role"], content=item["content"]))
    return out


@dataclass(frozen=True, slots=True)
class ChatTemplate:
    """Renders a conversation into the ChatML string the model expects."""

    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "ChatTemplate":
        """Load and sanity-check against the model's declared template.

        The default system prompt is read back out of the Jinja source rather
        than trusted from this file, so a model shipping a different one is a
        loud failure instead of a silent behaviour change.
        """
        config_path = Path(model_dir) / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        template = config.get("chat_template")
        if not template:
            raise UnsupportedConversationError(
                f"{config_path.name} declares no chat_template"
            )
        if DEFAULT_SYSTEM_PROMPT not in template:
            raise UnsupportedConversationError(
                "this model's chat template uses a different default system "
                "prompt than the one implemented here"
            )
        return cls()

    def render(
        self,
        messages: Sequence[Message | dict],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format a conversation, mirroring the model's Jinja template.

        Two details that the template makes easy to miss:

        * A system message is emitted from the *head* of the template, not
          from the message loop, and the loop then skips index 0. A system
          message appearing later in the conversation is emitted normally.
        * When no system message is supplied, the default one is inserted
          anyway. There is no such thing as a Qwen2.5 prompt without a system
          turn.
        """
        items = _coerce(messages)

        parts: list[str] = []

        if items and items[0].role == "system":
            parts.append(f"{IM_START}system\n{items[0].content}{IM_END}\n")
        else:
            parts.append(
                f"{IM_START}system\n{self.default_system_prompt}{IM_END}\n"
            )

        for index, message in enumerate(items):
            if message.role == "system" and index == 0:
                continue  # already emitted above
            parts.append(
                f"{IM_START}{message.role}\n{message.content}{IM_END}\n"
            )

        if add_generation_prompt:
            parts.append(f"{IM_START}assistant\n")

        return "".join(parts)


def encode_chat(
    tokenizer,
    messages: Sequence[Message | dict],
    add_generation_prompt: bool = True,
    template: ChatTemplate | None = None,
) -> list[int]:
    """Render a conversation and encode it to token IDs.

    ``split_added_tokens`` stays on, because the control tokens the template
    emits must become their own IDs. Note the consequence: a user message
    containing the literal text ``<|im_end|>`` will close the turn. Guarding
    against that belongs at the application layer, and the tokenizer already
    exposes ``split_added_tokens=False`` for encoding untrusted text on its own.
    """
    template = template or ChatTemplate()
    return tokenizer.encode(template.render(messages, add_generation_prompt))
