"""A deterministic corpus generator for tokenizer differential testing.

Ten thousand strings drawn from one seed, so a failure is reproducible and a
passing run means the same thing tomorrow. The categories are chosen for where
byte-level BPE actually breaks, not for realism:

* **Whitespace torture** -- the pre-token pattern has four separate whitespace
  alternatives and a lookahead. Most mismatches live here.
* **Random codepoints** -- the only category that can catch a disagreement
  between Python's Unicode tables and the reference's, since those differ only
  on characters assigned in versions the two do not share.
* **Combining marks** -- exercises the NFC normalizer, where skipping
  normalization changes the bytes and therefore the IDs.
* **Long repeated runs** -- exercises BPE tie-breaking, which is invisible on
  ordinary prose.
* **Control-token lookalikes** -- ``<|im_start|>`` and near-misses, to check
  added-token extraction and the longest-match rule.

Realistic prose is included too, but it is the category least likely to fail.
"""

from __future__ import annotations

import random
import unicodedata

WORDS = (
    "the of and to in a is that it for on with as was at by an be this from or "
    "have has had not but they we you he she its his her their our can will "
    "would should could about after all also any because been before between "
    "both during each few more most other over same some such than then there "
    "these those through under until very what when where which while who why "
    "model tensor vector matrix kernel cache token weight layer attention "
    "inference quantize embedding logits sample decode encode throughput"
).split()

CODE_SNIPPETS = [
    "def f(x):\n    return x + 1\n",
    "for i in range(10):\n    print(i)\n",
    "if (a && b) || !c { return 0; }",
    'const x = {"a": 1, "b": [2, 3]};',
    "SELECT * FROM users WHERE id = 42;",
    "#include <stdio.h>\nint main(void) { return 0; }",
    "let mut v: Vec<u8> = Vec::with_capacity(256);",
    "impl<T: Clone> Trait for Struct<T> {}",
    "x = np.zeros((3, 4), dtype=np.float32)",
    "git commit -m 'fix: off-by-one in kv cache'",
    "</div>\n<span class='x'>text</span>",
    "async fn main() -> Result<(), Box<dyn Error>> {}",
]

SCRIPTS = {
    "cjk": "日本語中文漢字テキストです繁體字简体字ひらがなカタカナ",
    "cyrillic": "Привет мир это тест кириллицы съешь ещё этих мягких",
    "arabic": "مرحبا بالعالم هذا اختبار للنص العربي",
    "hebrew": "שלום עולם זהו מבחן של טקסט עברי",
    "devanagari": "नमस्ते दुनिया यह एक परीक्षण है",
    "greek": "Γειά σου κόσμε αυτό είναι δοκιμή",
    "thai": "สวัสดีชาวโลกนี่คือการทดสอบ",
    "korean": "안녕하세요 세계 이것은 테스트입니다",
    "hangul_jamo": "ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉ",
}

EMOJI = "🦀🔥🎉😀😂🤔👍🚀💡🌍🐍⚡🧠📦🔧🫎🯰🥸🧿🜁"

PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~«»“”‘’—–…·§¶†‡"

CONTROL_TOKEN_LOOKALIKES = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|im_start|",          # near miss: unterminated
    "<|im_start|>>",        # near miss: trailing
    "< |im_start| >",       # near miss: spaced
    "<|IM_START|>",         # near miss: case
    "<|vision_start|><|vision_end|>",
]

COMBINING = "\u0301\u0302\u0303\u0308\u030a\u0327\u0331"

WHITESPACE = [" ", "  ", "\t", "\n", "\r\n", "\n\n", " \n ", "\u00a0", "\u3000", "\u2009"]


def _sentence(rng: random.Random) -> str:
    n = rng.randint(1, 14)
    words = [rng.choice(WORDS) for _ in range(n)]
    if rng.random() < 0.3:
        words[0] = words[0].capitalize()
    text = " ".join(words)
    if rng.random() < 0.5:
        text += rng.choice(".?!,;:")
    return text


def _contractions(rng: random.Random) -> str:
    base = ["don't", "can't", "I'll", "we've", "they're", "it's", "he'd", "you'm"]
    picked = [rng.choice(base) for _ in range(rng.randint(1, 5))]
    if rng.random() < 0.3:
        picked = [p.upper() for p in picked]
    return " ".join(picked)


def _whitespace_torture(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(2, 8)):
        parts.append(rng.choice(WHITESPACE))
        if rng.random() < 0.6:
            parts.append(rng.choice(WORDS))
    return "".join(parts)


def _numbers(rng: random.Random) -> str:
    style = rng.randint(0, 5)
    if style == 0:
        return str(rng.randint(0, 10**rng.randint(1, 12)))
    if style == 1:
        return f"{rng.uniform(-1e6, 1e6):.{rng.randint(0, 8)}f}"
    if style == 2:
        return f"{rng.randint(1900, 2100)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if style == 3:
        return f"{rng.randint(0, 999)},{rng.randint(0, 999):03d},{rng.randint(0, 999):03d}"
    if style == 4:
        return f"0x{rng.randrange(16**8):08x}"
    return "".join(rng.choice("0123456789٣٤۵๗½¾Ⅷ") for _ in range(rng.randint(1, 20)))


def _multilingual(rng: random.Random) -> str:
    script = SCRIPTS[rng.choice(list(SCRIPTS))]
    start = rng.randrange(len(script))
    chunk = script[start : start + rng.randint(1, 15)]
    if rng.random() < 0.4:
        chunk = f"{rng.choice(WORDS)} {chunk} {rng.choice(WORDS)}"
    return chunk


def _emoji(rng: random.Random) -> str:
    out = "".join(rng.choice(EMOJI) for _ in range(rng.randint(1, 6)))
    if rng.random() < 0.3:
        out = f"{rng.choice(WORDS)} {out} {rng.choice(WORDS)}"
    if rng.random() < 0.15:
        out += "\u200d".join(rng.choice(EMOJI) for _ in range(2))
    return out


def _random_codepoints(rng: random.Random) -> str:
    """Random assigned codepoints, including well outside the BMP.

    Surrogates are excluded because they cannot be encoded as UTF-8 at all.
    Unassigned codepoints are skipped: the two implementations may legitimately
    have been built against different Unicode versions, and the point of this
    category is to find real disagreements, not to litigate which version each
    side was compiled with.
    """
    chars = []
    while len(chars) < rng.randint(1, 30):
        cp = rng.randrange(0x110000)
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        if unicodedata.category(ch) == "Cn":
            continue
        chars.append(ch)
    return "".join(chars)


def _combining(rng: random.Random) -> str:
    """Strings that NFC will change, to exercise the normalizer."""
    out = []
    for _ in range(rng.randint(1, 10)):
        out.append(rng.choice("aeiounycAEIOU"))
        for _ in range(rng.randint(1, 3)):
            out.append(rng.choice(COMBINING))
    return "".join(out)


def _repeats(rng: random.Random) -> str:
    """Long runs, where BPE tie-breaking becomes observable."""
    unit = rng.choice(["a", "ab", " ", "  ", "\n", "0", "=-", "ha", "日", "\t"])
    return unit * rng.randint(2, 60)


def _punctuation(rng: random.Random) -> str:
    return "".join(rng.choice(PUNCT) for _ in range(rng.randint(1, 25)))


def _urls(rng: random.Random) -> str:
    host = rng.choice(["example.com", "a.b.co.uk", "localhost:8080", "127.0.0.1"])
    path = "/".join(rng.choice(WORDS) for _ in range(rng.randint(0, 4)))
    tail = f"?q={rng.randint(0, 999)}&r={rng.choice(WORDS)}" if rng.random() < 0.5 else ""
    kind = rng.random()
    if kind < 0.4:
        return f"https://{host}/{path}{tail}"
    if kind < 0.7:
        return f"{rng.choice(WORDS)}@{host}"
    return rf"C:\Users\{rng.choice(WORDS)}\{path}"


def _control_tokens(rng: random.Random) -> str:
    token = rng.choice(CONTROL_TOKEN_LOOKALIKES)
    if rng.random() < 0.5:
        return f"{_sentence(rng)}{token}{_sentence(rng)}"
    return token


def _latin1_noise(rng: random.Random) -> str:
    """Arbitrary bytes read as latin-1: dense in the 0x80-0xFF range."""
    raw = bytes(rng.randrange(256) for _ in range(rng.randint(1, 40)))
    return raw.decode("latin-1")


def _mixed(rng: random.Random) -> str:
    makers = [_sentence, _numbers, _multilingual, _emoji, _punctuation, _whitespace_torture]
    return "".join(rng.choice(makers)(rng) for _ in range(rng.randint(2, 5)))


GENERATORS = [
    _sentence,
    _contractions,
    _whitespace_torture,
    _numbers,
    _multilingual,
    _emoji,
    _random_codepoints,
    _combining,
    _repeats,
    _punctuation,
    _urls,
    _control_tokens,
    _latin1_noise,
    _mixed,
]


def generate(count: int = 10_000, seed: int = 20260914) -> list[str]:
    """Produce ``count`` strings, cycling the categories so each is covered."""
    rng = random.Random(seed)
    strings: list[str] = []

    # A few fixed cases first, so the corpus always contains the degenerate
    # inputs a random generator would rarely produce.
    strings.extend(["", " ", "\n", "\t", "a", "\x00", "\x1c", "\U0001FACE", "é", "é"])

    while len(strings) < count:
        maker = GENERATORS[len(strings) % len(GENERATORS)]
        strings.append(maker(rng))

    return strings[:count]


def category_of(index: int) -> str:
    """Which generator produced the string at this index, for error messages."""
    if index < 10:
        return "fixed"
    return GENERATORS[index % len(GENERATORS)].__name__.lstrip("_")


if __name__ == "__main__":
    import collections

    items = generate()
    print(f"{len(items)} strings")
    print(f"total characters: {sum(len(s) for s in items):,}")
    lengths = collections.Counter(len(s) // 20 * 20 for s in items)
    print("length histogram (bucket -> count):")
    for bucket in sorted(lengths)[:10]:
        print(f"  {bucket:4d}+ {lengths[bucket]}")
