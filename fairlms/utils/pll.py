"""Pseudo-log-likelihood scoring helpers (AUL, AULA, CAT, CPS, PLL)."""

from __future__ import annotations

import difflib

import torch


def get_token_ranks(log_probs, token_ids):
    ranks = []
    for i in range(log_probs.shape[0]):
        sorted_indices = torch.argsort(log_probs[i], descending=True)
        gold_id = token_ids[i].item()
        rank = (sorted_indices == gold_id).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    return ranks


def score_sentence(tokenizer, model, sentence, use_attention=False):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits.squeeze(0)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_ids = input_ids.view(-1, 1).detach()
        token_log_probs = log_probs.gather(1, token_ids)[1:-1].squeeze(1)

        if use_attention:
            # Prefer a generic encoder attribute when present; fall back to BERT.
            encoder = getattr(model, "bert", None) or getattr(model, "roberta", None)
            if encoder is None:
                raise AttributeError(
                    "use_attention=True requires a model with .bert or .roberta"
                )
            # SDPA/flash attention backends silently drop attentions even when
            # output_attentions=True; force eager attention to get real weights.
            if getattr(model.config, "_attn_implementation", None) != "eager" and hasattr(
                model, "set_attn_implementation"
            ):
                model.set_attn_implementation("eager")
            enc_output = encoder(input_ids=input_ids, output_attentions=True)
            all_attentions = torch.stack(
                [a.squeeze(0) for a in enc_output.attentions], dim=0
            )
            mean_attention = all_attentions.mean(dim=(0, 1))
            token_attention = mean_attention.mean(dim=0)
            token_log_probs = token_log_probs * token_attention[1:-1]

    score = torch.mean(token_log_probs).item()
    ranks = get_token_ranks(log_probs[1:-1], token_ids[1:-1])
    return score, ranks


def get_span(tokens1, tokens2, operation):
    tokens1 = [str(x) for x in tokens1.tolist()]
    tokens2 = [str(x) for x in tokens2.tolist()]

    matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        if (operation == "equal" and op[0] == "equal") or (
            operation == "diff" and op[0] != "equal"
        ):
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2
