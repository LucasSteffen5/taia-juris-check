# -*- coding: utf-8 -*-
"""Windowed NER inference with original Unicode character offsets."""
from typing import Any

from ..config import NER_LABELS, WINDOW_OVERLAP, WINDOW_SIZE


def extract_ner_citations(
    text: str,
    model: Any,
    tokenizer: Any,
    window_size: int = WINDOW_SIZE,
    overlap: int = WINDOW_OVERLAP,
) -> list[dict[str, Any]]:
    import numpy as np
    import torch

    device = next(model.parameters()).device
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=False,
    )
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    if not input_ids:
        return []

    try:
        special_count = tokenizer.num_special_tokens_to_add(pair=False)
    except Exception:
        special_count = 2
    payload_width = window_size - special_count
    stride = payload_width - overlap
    if payload_width <= 0 or stride <= 0:
        raise ValueError("A janela precisa comportar tokens e manter avanço positivo.")

    id2label = getattr(model.config, "id2label", None)
    if id2label:
        num_labels = len(id2label)
        label_names = {int(k): v for k, v in id2label.items()}
    else:
        num_labels = len(NER_LABELS)
        label_names = {i: l for i, l in enumerate(NER_LABELS)}

    probabilities = np.zeros((len(input_ids), num_labels), dtype=np.float32)
    counts = np.zeros((len(input_ids),), dtype=np.float32)
    cls_id = getattr(tokenizer, "cls_token_id", 101) or 101
    sep_id = getattr(tokenizer, "sep_token_id", 102) or 102

    for start in range(0, len(input_ids), stride):
        end = min(start + payload_width, len(input_ids))
        body_ids = input_ids[start:end]
        try:
            model_ids = tokenizer.build_inputs_with_special_tokens(body_ids)
            special_mask = tokenizer.get_special_tokens_mask(body_ids, already_has_special_tokens=False)
            body_positions = [index for index, is_special in enumerate(special_mask) if not is_special]
        except Exception:
            model_ids = [cls_id] + body_ids + [sep_id]
            body_positions = list(range(1, len(body_ids) + 1))
        if len(body_positions) != len(body_ids):
            raise ValueError("Tokenizer não preservou o alinhamento dos tokens da janela.")
        ids_tensor = torch.tensor([model_ids], dtype=torch.long, device=device)
        mask_tensor = torch.ones_like(ids_tensor)
        with torch.no_grad():
            output = model(input_ids=ids_tensor, attention_mask=mask_tensor)
            logits = output.logits[0, body_positions].float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probabilities[start:end] += probs
        counts[start:end] += 1.0
        if end == len(input_ids):
            break

    probabilities /= np.maximum(counts[:, None], 1.0)
    pred_ids = np.argmax(probabilities, axis=-1)

    import re
    from .distractor_filter import is_distractor_span, find_header_boundary

    vague_head_re = re.compile(
        r"\b(?:julgado|ac[óo]rd[ãa]o|precedente|decis[ãa]o|recurso(?:\s+em\s+habeas\s+corpus)?|reclama[çc][ãa]o|rcl|apl|agravo(?:\s+em\s+recurso\s+especial)?)\b",
        re.IGNORECASE
    )

    # 1. Agrupar entidades básicas do BERT com subwords e contíguos
    raw_entities: list[dict[str, Any]] = []
    curr: dict[str, Any] | None = None

    for index, tag_id in enumerate(pred_ids):
        start, end = offsets[index]
        if start == end:
            continue
        tag = label_names.get(int(tag_id), "O")
        token_str = tokenizer.decode([input_ids[index]]) if hasattr(tokenizer, "decode") else ""
        is_subword = token_str.startswith("##")
        prob = float(probabilities[index, tag_id])

        if tag.startswith("B-") or tag.startswith("I-"):
            ent_type = tag[2:]
            if curr and curr["type"] == ent_type:
                gap = text[curr["end"]:start]
                if tag.startswith("I-") or is_subword or (gap in ("", " ", ".", "-", "/")):
                    curr["end"] = end
                    curr["probs"].append(prob)
                    continue
            if curr:
                raw_entities.append(curr)
            curr = {
                "start": start,
                "end": end,
                "type": ent_type,
                "probs": [prob],
            }
        else:
            if curr:
                raw_entities.append(curr)
                curr = None
    if curr:
        raw_entities.append(curr)

    for e in raw_entities:
        e["prob"] = float(np.mean(e.pop("probs")))
        e["text"] = text[e["start"]:e["end"]]

    # 2. Composição Semântica e Boundary Snapping
    candidates: list[dict[str, Any]] = []
    header_end = find_header_boundary(text)

    # A. Citações Formais Diretas (JURISPRUDENCIA e LEGISLACAO)
    for ent in raw_entities:
        is_jur = "JUR" in ent["type"]
        is_lei = "LEI" in ent["type"] or "LEG" in ent["type"]
        if is_jur or is_lei:
            c_type = "jurisprudencia" if is_jur else "lei"
            start_pos = ent["start"]
            end_pos = ent["end"]
            is_vague = False

            # Extensão de Jurisprudência: verificar se segue TEMPO e/ou PESSOA
            if c_type == "jurisprudencia":
                for next_ent in raw_entities:
                    if next_ent["start"] > end_pos and (next_ent["start"] - end_pos) < 60:
                        bridge = text[end_pos:next_ent["start"]].lower()
                        if next_ent["type"] in ("TEMPO", "PESSOA") and any(
                            b in bridge for b in ("de", "em", "rel.", "relator", "relatoria", "min.")
                        ):
                            end_pos = next_ent["end"]
                            is_vague = True

            # Boundary Snapping para Leis: estender até número se parou antes dele
            if c_type == "lei":
                after_match = re.match(r"^(\s*(?:n[ºo°\.]\s*)?[\d\.]+(?:/\d{4})?)", text[end_pos:], re.IGNORECASE)
                if after_match and not re.search(r"\d", ent["text"]):
                    end_pos += after_match.end()
                if not re.search(r"\d", text[start_pos:end_pos]):
                    continue

            candidates.append({
                "inicio": start_pos,
                "fim": end_pos,
                "tipo": c_type,
                "prob": ent["prob"],
                "is_vague": is_vague,
            })

    # B. Citações Vagas com Âncoras Nominais Judiciais (ex: "julgado do STF...")
    for m in vague_head_re.finditer(text):
        h_start, h_end = m.start(), m.end()
        matching_org = [
            e for e in raw_entities
            if ("ORG" in e["type"] or "JUR" in e["type"])
            and e["start"] >= h_end and (e["start"] - h_end) < 25
        ]
        if matching_org:
            org = matching_org[0]
            curr_end = org["end"]
            matching_rel = [
                e for e in raw_entities
                if e["type"] in ("TEMPO", "PESSOA")
                and e["start"] > curr_end and (e["start"] - curr_end) < 60
            ]
            if matching_rel:
                last_ent = matching_rel[-1]
                bridge = text[curr_end:last_ent["start"]].lower()
                if any(b in bridge for b in ("de", "em", "rel", "relator", "relatoria", "proferido", "julgado")):
                    candidates.append({
                        "inicio": h_start,
                        "fim": last_ent["end"],
                        "tipo": "jurisprudencia",
                        "prob": 0.95,
                        "is_vague": True,
                    })

    # 3. Filtrar distratores e desduplicar por maior cobertura de span
    valid_cits: list[dict[str, Any]] = []
    for c in candidates:
        if not is_distractor_span(text, c["inicio"], c["fim"], header_end, is_vague=c["is_vague"]):
            c["trecho"] = text[c["inicio"]:c["fim"]]
            valid_cits.append(c)

    valid_cits.sort(key=lambda x: (x["inicio"], -(x["fim"] - x["inicio"])))
    deduped: list[dict[str, Any]] = []
    for cand in valid_cits:
        if not deduped:
            deduped.append(cand)
            continue
        last = deduped[-1]
        if cand["inicio"] < last["fim"]:
            if (cand["fim"] - cand["inicio"]) > (last["fim"] - last["inicio"]):
                deduped[-1] = cand
        else:
            deduped.append(cand)

    return deduped
