"""A1111-style prompt encoding.

Supports (word:1.3), (word) = x1.1, [word] = /1.1, \\( escapes, BREAK, prompts longer
than 77 tokens (encoded in 75-token chunks and concatenated), clip skip, and
textual-inversion tokens.
"""

import re

import torch

CHUNK = 75  # 77 minus BOS/EOS

_re_attention = re.compile(
    r"\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:\s*([+-]?[.\d]+)\s*\)|\)|]|[^\\()\[\]:]+|:"
)
_re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text: str) -> list[list]:
    """Split a prompt into [text, weight] pairs; weight -1 marks a BREAK."""
    res: list[list] = []
    round_brackets: list[int] = []
    square_brackets: list[int] = []

    def multiply_range(start: int, mult: float):
        for p in range(start, len(res)):
            res[p][1] *= mult

    for m in _re_attention.finditer(text):
        token, weight = m.group(0), m.group(1)
        if token.startswith("\\"):
            res.append([token[1:], 1.0])
        elif token == "(":
            round_brackets.append(len(res))
        elif token == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif token == ")" and round_brackets:
            multiply_range(round_brackets.pop(), 1.1)
        elif token == "]" and square_brackets:
            multiply_range(square_brackets.pop(), 1 / 1.1)
        else:
            for i, part in enumerate(_re_break.split(token)):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, 1.1)
    for pos in square_brackets:
        multiply_range(pos, 1 / 1.1)

    if not res:
        res = [["", 1.0]]
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1] and res[i][1] != -1:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res


def _chunks(pipe, text: str) -> list[tuple[list[int], list[float]]]:
    tokenizer = pipe.tokenizer
    chunks, tokens, weights = [], [], []

    def flush():
        nonlocal tokens, weights
        pad = CHUNK - len(tokens)
        chunks.append((tokens + [tokenizer.eos_token_id] * pad, weights + [1.0] * pad))
        tokens, weights = [], []

    for fragment, weight in parse_prompt_attention(text):
        if weight == -1:
            flush()
            continue
        # expands multi-vector textual-inversion tokens (name -> name name_1 ...)
        fragment = pipe.maybe_convert_prompt(fragment, tokenizer)
        for t in tokenizer(fragment, add_special_tokens=False, truncation=False).input_ids:
            if len(tokens) == CHUNK:
                flush()
            tokens.append(t)
            weights.append(weight)
    if tokens or not chunks:
        flush()
    return chunks


def _encode(pipe, chunks, clip_skip: int, device) -> torch.Tensor:
    te, tokenizer = pipe.text_encoder, pipe.tokenizer
    # transformers 5 flattened CLIPTextModel; older versions nest it under .text_model
    final_norm = te.text_model.final_layer_norm if hasattr(te, "text_model") else te.final_layer_norm
    out = []
    for tokens, weights in chunks:
        ids = torch.tensor([[tokenizer.bos_token_id, *tokens, tokenizer.eos_token_id]], device=device)
        w = torch.tensor([1.0, *weights, 1.0], device=device, dtype=te.dtype)
        o = te(ids, output_hidden_states=clip_skip > 1)
        if clip_skip > 1:
            h = final_norm(o.hidden_states[-clip_skip])
        else:
            h = o.last_hidden_state
        # A1111 weighting: scale token embeddings, then restore the original mean
        mean = h.mean()
        h = h * w[None, :, None]
        h = h * (mean / h.mean())
        out.append(h)
    return torch.cat(out, dim=1)


def encode_prompt_pair(pipe, prompt: str, negative: str, clip_skip: int, device):
    """clip_skip uses A1111 numbering: 1 = last layer (no skip), 2 = penultimate."""
    pos, neg = _chunks(pipe, prompt), _chunks(pipe, negative)
    empty = ([pipe.tokenizer.eos_token_id] * CHUNK, [1.0] * CHUNK)
    while len(pos) < len(neg):
        pos.append(empty)
    while len(neg) < len(pos):
        neg.append(empty)
    return _encode(pipe, pos, clip_skip, device), _encode(pipe, neg, clip_skip, device)
