"""Prepare audited, group-disjoint token-classification data without editing sources."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import (
    AUXILIARY_CONFIDENCE_THRESHOLD,
    AUXILIARY_LABELS,
    NER_LABELS,
    OFFICIAL_TRAIN_DOCUMENTS,
    OFFICIAL_REPEAT,
    SEED,
    WINDOW_OVERLAP,
    WINDOW_SIZE,
)
from .provenance import sha256_file, sha256_tree, write_json


@dataclass
class PreparedDocument:
    document_id: str
    source: str
    split: str
    level: int
    text: str
    annotations: list[dict[str, Any]]
    ignored_regions: list[tuple[int, int, str]]
    raw_file_sha256: str
    ignored_annotations: list[dict[str, Any]] = field(default_factory=list)
    negative_regions: list[tuple[int, int, str]] = field(default_factory=list)
    teacher_label_repairs: list[dict[str, Any]] = field(default_factory=list)
    base_document_id: str | None = None
    augmentation_rate: float = 0.0


def _read_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return text, digest


def _read_annotations(path: Path, document_ids: set[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "nivel", "documento_id", "citacao_id", "inicio", "fim", "trecho",
            "tipo", "classificacao", "id_canonico",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Colunas obrigatórias ausentes em {path}.")
        rows = []
        for row in reader:
            if document_ids is not None and row["documento_id"] not in document_ids:
                continue
            rows.append(row)
        return rows


def _normalize_annotation_surface(value: str) -> str:
    """Some official CSV rows store line breaks as literal backslash escapes."""
    return value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")


def _deterministic_context_split(context_key: str) -> str:
    return hashlib.sha256(f"{SEED}:{context_key}".encode("utf-8")).hexdigest()


def _local_document_groups(
    local_text_dir: Path,
    annotated_document_ids: set[str] | None = None,
) -> tuple[dict[str, list[Path]], dict[str, str]]:
    docs_by_group: dict[str, list[Path]] = defaultdict(list)
    fixed_splits: dict[str, str] = {}
    files = sorted(local_text_dir.glob("*.txt"))
    if len(files) != 530:
        raise ValueError(f"Esperados 530 textos locais; encontrados {len(files)}.")

    synthetic_pattern = re.compile(r"^syn_n([12])_(\d{6})$")
    lener_pattern = re.compile(r"^lener_(train|dev|test)_n([12])_(.+)$")
    for path in files:
        syn = synthetic_pattern.fullmatch(path.stem)
        if syn:
            context_no = (int(syn.group(2)) - 1) // 3 + 1
            docs_by_group[f"synthetic_{context_no:03d}"].append(path)
            continue
        lener = lener_pattern.fullmatch(path.stem)
        if not lener:
            raise ValueError(f"Nome de documento local fora dos padrões esperados: {path.name}.")
        source_split, _level, case_id = lener.groups()
        group_key = f"lener_{case_id}"
        docs_by_group[group_key].append(path)
        normalized_split = {"train": "train", "dev": "validation", "test": "test"}[source_split]
        existing = fixed_splits.setdefault(group_key, normalized_split)
        if existing != normalized_split:
            raise ValueError(f"O mesmo processo LeNER está em duas divisões: {group_key}.")

    synthetic_groups = {key: value for key, value in docs_by_group.items() if key.startswith("synthetic_")}
    if len(synthetic_groups) != 65 or any(len(paths) != 6 for paths in synthetic_groups.values()):
        raise ValueError("Os 65 contextos sintéticos devem ter seis variantes indivisíveis cada.")
    for key, paths in synthetic_groups.items():
        variants = Counter(re.fullmatch(r"^syn_n([12])_(\d{6})$", p.stem).group(1) for p in paths)
        if variants != Counter({"1": 3, "2": 3}):
            raise ValueError(f"Variantes sintéticas incompletas no contexto {key}.")

    ordered_groups = sorted(synthetic_groups, key=_deterministic_context_split)
    train_groups = set(ordered_groups[:52])
    validation_groups = set(ordered_groups[52:59])
    test_groups = set(ordered_groups[59:])
    if (len(train_groups), len(validation_groups), len(test_groups)) != (52, 7, 6):
        raise AssertionError("A partição sintética seed 42 não resultou em 52/7/6 grupos.")

    splits: dict[str, str] = {}
    for group_key in synthetic_groups:
        splits[group_key] = (
            "train" if group_key in train_groups else
            "validation" if group_key in validation_groups else "test"
        )

    # The local corpus contains three fully negative synthetic contexts. The hash
    # split previously put all three in training, leaving validation and test
    # unable to measure false positives on documents with no citations. Keep the
    # 52/7/6 group counts and swap complete context groups so every split has a
    # held-out negative context, while retaining at least one negative in train.
    if annotated_document_ids is not None:
        negative_groups = {
            group_key
            for group_key, paths in synthetic_groups.items()
            if all(path.stem not in annotated_document_ids for path in paths)
        }
        if len(negative_groups) >= 3:
            split_by_group = dict(splits)
            for target in ("validation", "test"):
                if any(split_by_group[group] == target for group in negative_groups):
                    continue
                donor_counts = Counter(
                    split_by_group[group] for group in negative_groups
                )
                donor_candidates = [
                    group for group in negative_groups
                    if split_by_group[group] != target
                    and (split_by_group[group] != "train" or donor_counts["train"] > 1)
                ]
                donor_group = min(
                    donor_candidates,
                    key=lambda group: (
                        0 if split_by_group[group] == "train" else 1,
                        _deterministic_context_split(group),
                    ),
                )
                replacement_group = next(
                    group for group in sorted(synthetic_groups, key=_deterministic_context_split)
                    if split_by_group[group] == target and group not in negative_groups
                )
                donor_split = split_by_group[donor_group]
                split_by_group[donor_group] = target
                split_by_group[replacement_group] = donor_split
            if not any(split_by_group[group] == "train" for group in negative_groups):
                raise ValueError("A realocação de negativos removeu todos os negativos do treino.")
            splits = split_by_group
    splits.update(fixed_splits)

    expected_lener_counts = {"train": 100, "validation": 20, "test": 20}
    lener_counts = Counter()
    for group_key, paths in docs_by_group.items():
        if not group_key.startswith("lener_"):
            continue
        members = [re.fullmatch(r"^lener_(train|dev|test)_n([12])_(.+)$", p.stem) for p in paths]
        levels = Counter(member.group(2) for member in members if member)
        if levels != Counter({"1": 1, "2": 1}):
            raise ValueError(f"Cada processo LeNER deve ter versões N1 e N2: {group_key}.")
        lener_counts[fixed_splits[group_key]] += len(paths)
    if dict(lener_counts) != expected_lener_counts:
        raise ValueError(f"Divisões LeNER inesperadas: {dict(lener_counts)}.")

    assignments = {
        path.stem: splits[
            f"synthetic_{(int(synthetic_pattern.fullmatch(path.stem).group(2)) - 1) // 3 + 1:03d}"
            if synthetic_pattern.fullmatch(path.stem)
            else f"lener_{lener_pattern.fullmatch(path.stem).group(3)}"
        ]
        for path in files
    }
    if Counter(assignments.values()) != Counter({"train": 412, "validation": 62, "test": 56}):
        raise ValueError(f"Divisões locais inesperadas: {dict(Counter(assignments.values()))}.")
    return {}, assignments


def _context_split_map(
    local_text_dir: Path,
    local_annotation_path: Path | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Assign every document to a split and retain its non-leaking context key."""
    annotated_document_ids = None
    if local_annotation_path is not None:
        annotated_document_ids = {
            row["documento_id"] for row in _read_annotations(local_annotation_path)
        }
    _unused, assignments = _local_document_groups(local_text_dir, annotated_document_ids)
    groups: dict[str, str] = {}
    for path in sorted(local_text_dir.glob("*.txt")):
        syn = re.fullmatch(r"^syn_n([12])_(\d{6})$", path.stem)
        if syn:
            groups[path.stem] = f"synthetic_{(int(syn.group(2)) - 1) // 3 + 1:03d}"
        else:
            lener = re.fullmatch(r"^lener_(train|dev|test)_n([12])_(.+)$", path.stem)
            assert lener
            groups[path.stem] = f"lener_{lener.group(3)}"
    if annotated_document_ids is not None:
        negative_groups = {
            group for group, members in _group_members(groups).items()
            if all(document_id not in annotated_document_ids for document_id in members)
            and group.startswith("synthetic_")
        }
        negative_group_splits = {
            split: {
                group for group in negative_groups
                if any(assignments[document_id] == split for document_id in groups if groups[document_id] == group)
            }
            for split in ("train", "validation", "test")
        }
        if len(negative_groups) >= 3 and any(not rows for rows in negative_group_splits.values()):
            raise ValueError("Contextos negativos devem estar presentes em treino, validação e teste.")
    return assignments, groups


def _group_members(document_groups: dict[str, str]) -> dict[str, list[str]]:
    members: dict[str, list[str]] = defaultdict(list)
    for document_id, group in document_groups.items():
        members[group].append(document_id)
    return members


def _document_sampling_group(document_id: str, source: str) -> str:
    if source == "official":
        return document_id
    synthetic = re.fullmatch(r"syn_n[12]_(\d{6})(?:__ocr_\d+)?", document_id)
    if synthetic:
        return f"synthetic_{(int(synthetic.group(1)) - 1) // 3 + 1:03d}"
    lener = re.fullmatch(r"lener_(?:train|dev|test)_n[12]_(.+?)(?:__ocr_\d+)?", document_id)
    return f"lener_{lener.group(1)}" if lener else document_id


def _make_ocr_variant(document: PreparedDocument, noise_rate: float, seed: int) -> tuple[PreparedDocument, list[dict[str, Any]]]:
    import random
    from dataclasses import replace

    if not 0.0 < noise_rate < 1.0:
        raise ValueError("A taxa de ruído OCR precisa ficar entre 0 e 1.")
    substitutions = {
        "o": ("0",), "0": ("o",), "i": ("1",), "l": ("1",),
        "1": ("l", "i"), "s": ("5",), "5": ("s",), "b": ("8",), "8": ("b",),
    }
    eligible = [index for index, char in enumerate(document.text) if char.lower() in substitutions]
    target_count = round(len(eligible) * noise_rate)
    generator = random.Random(seed)
    changed = sorted(generator.sample(eligible, min(target_count, len(eligible)))) if target_count else []
    chars = list(document.text)
    repairs = []
    for index in changed:
        before = chars[index]
        choices = substitutions[before.lower()]
        after = generator.choice(choices)
        if before.isupper():
            after = after.upper()
        chars[index] = after
        repairs.append({"offset": index, "from": before, "to": after})
    variant_id = f"{document.document_id}__ocr_{int(noise_rate * 100):02d}"
    variant = replace(
        document,
        document_id=variant_id,
        text="".join(chars),
        raw_file_sha256=hashlib.sha256("".join(chars).encode("utf-8")).hexdigest(),
        annotations=[dict(row) for row in document.annotations],
        ignored_regions=list(document.ignored_regions),
        negative_regions=list(document.negative_regions),
        ignored_annotations=[dict(row) for row in document.ignored_annotations],
        teacher_label_repairs=[],
        base_document_id=document.document_id,
        augmentation_rate=noise_rate,
    )
    if len(variant.text) != len(document.text):
        raise AssertionError("A transformação OCR alterou o comprimento e invalidou offsets.")
    for annotation in variant.annotations:
        start, end = int(annotation["start"]), int(annotation["end"])
        if variant.text[start:end] and len(variant.text[start:end]) != end - start:
            raise AssertionError("Remapeamento de offset OCR inconsistente.")
    return variant, repairs


def _official_selection() -> tuple[str, ...]:
    ranked: list[str] = []
    for level in (1, 2):
        ids = [f"gen_n{level}_{index:03d}" for index in range(1, 14)]
        ranked.extend(sorted(ids, key=lambda doc: hashlib.sha256(f"{SEED}:{doc}".encode()).hexdigest())[:5])
    selected = tuple(sorted(ranked))
    if selected != tuple(sorted(OFFICIAL_TRAIN_DOCUMENTS)):
        raise ValueError("A seleção oficial seed 42 diverge da exceção autorizada no AGENTS.md.")
    return selected


def _header_boundary(text: str) -> int:
    match = re.search(
        r"\n\s*(?:I\s*[-—–]|EMENTA\b|RELAT[ÓO]RIO\b|MEMORIAL\b|CONTRARRAZ[ÕO]ES\b|PARECER\b)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.start()
    # Without an explicit section marker, do not infer a header from arbitrary lines.
    return 0


def _confirmed_header_distractor(text: str, start: int, end: int, header_end: int) -> str | None:
    if start >= header_end:
        return None
    before = text[max(0, start - 150):start]
    surface = text[start:end]
    checks = (
        (r"(?:n[ºo°.]?\s*)?processo\s*:\s*(?:[^\n]{0,120})$", "numero_do_processo_no_cabecalho"),
        (r"feito\s+principal\s*\([^\n)]*$", "feito_principal_no_cabecalho"),
        (r"ac[oó]rd[aã]o\s+n\s*\.??\s*:?\s*$", "numero_interno_do_acordao"),
        (r"(?:protocolo|identificador\s+interno)\s*(?:n[ºo°.]\s*)?[:#]?\s*$", "protocolo_ou_identificador_interno"),
        (r"\boab\s*/\s*[a-z]{2}\s*$", "registro_oab"),
        (r"\bfls?\.?\s*$|\bfolhas?\s*$", "referencia_a_folhas"),
    )
    for pattern, reason in checks:
        if re.search(pattern, before, re.IGNORECASE):
            return reason
    if re.match(r"^(?:autos|processo|protocolo)\s*(?:n[ºo°.]|:)", surface.strip(), re.IGNORECASE):
        return "identificador_administrativo_no_cabecalho"
    return None


def _collect_documents(
    local_text_dir: Path,
    local_annotation_path: Path,
    official_text_dir: Path,
    official_annotation_path: Path,
) -> tuple[list[PreparedDocument], dict[str, Any]]:
    local_rows = _read_annotations(local_annotation_path)
    assignments, context_groups = _context_split_map(local_text_dir, local_annotation_path)
    selected_official = _official_selection()
    selected_set = set(selected_official)
    official_rows = _read_annotations(official_annotation_path, selected_set)
    observed_selected = {row["documento_id"] for row in official_rows}
    official_files = {doc_id: official_text_dir / f"{doc_id}.txt" for doc_id in selected_official}
    missing = [doc_id for doc_id, path in official_files.items() if not path.is_file()]
    if missing or observed_selected != selected_set:
        raise ValueError(f"Entradas de treino oficiais ausentes: arquivos={missing}; goldens sem linhas={sorted(selected_set-observed_selected)}.")

    by_doc: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in local_rows:
        if row["documento_id"] not in assignments:
            raise ValueError(f"Anotação local sem texto: {row['documento_id']}.")
        by_doc[row["documento_id"]].append(row)
    official_by_doc: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in official_rows:
        official_by_doc[row["documento_id"]].append(row)

    documents: list[PreparedDocument] = []
    transformations = Counter()
    for path in sorted(local_text_dir.glob("*.txt")):
        text, source_sha = _read_text(path)
        split = assignments[path.stem]
        documents.append(
            _make_document(path.stem, "local", split, text, source_sha, by_doc[path.stem], transformations)
        )
    for doc_id in selected_official:
        path = official_files[doc_id]
        text, source_sha = _read_text(path)
        documents.append(
            _make_document(doc_id, "official", "train", text, source_sha, official_by_doc[doc_id], transformations)
        )

    duplicate_hashes: dict[str, tuple[str, str]] = {}
    for document in documents:
        digest = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
        prior = duplicate_hashes.get(digest)
        if prior and (prior[0], prior[1]) != (document.source, document.split):
            raise ValueError(
                f"Texto idêntico atravessa divisões: {prior[1]}/{prior[0]} e {document.split}/{document.source}."
            )
        duplicate_hashes[digest] = (document.source, document.split)

    local_split_docs = Counter(doc.split for doc in documents if doc.source == "local")
    if local_split_docs != Counter({"train": 412, "validation": 62, "test": 56}):
        raise ValueError(f"Contagem de divisão local diverge: {dict(local_split_docs)}.")
    grouped_splits: dict[str, str] = {}
    for document in documents:
        if document.source == "local":
            group = context_groups[document.document_id]
            previous = grouped_splits.setdefault(group, document.split)
            if previous != document.split:
                raise ValueError(f"Vazamento entre divisões no grupo {group}.")

    local_unannotated_by_split = {
        split: sum(
            document.source == "local" and document.split == split and not by_doc.get(document.document_id)
            for document in documents
        )
        for split in ("train", "validation", "test")
    }
    negative_context_groups_by_split = {
        split: sorted({
            context_groups[document.document_id]
            for document in documents
            if document.source == "local" and document.split == split
            and document.document_id.startswith("syn_") and not by_doc.get(document.document_id)
        })
        for split in ("train", "validation", "test")
    }
    audit = {
        "seed": SEED,
        "local_source_files_sha256": sha256_tree(local_text_dir),
        "local_annotation_sha256": sha256_file(local_annotation_path),
        "official_train_source_files_sha256": _hash_selected_files(official_files),
        "official_annotation_sha256": sha256_file(official_annotation_path),
        "official_staging_manifest_sha256": (
            sha256_file(official_text_dir.parent / "sources_manifest.json")
            if (official_text_dir.parent / "sources_manifest.json").is_file() else None
        ),
        "official_annotation_rows_examined": len(official_rows),
        "local_annotation_rows_read": len(local_rows),
        "official_selected_annotation_rows_read": len(official_rows),
        "official_documents_authorized_for_training": list(selected_official),
        "official_documents_reserved_for_requested_final_score": 16,
        "official_document_ids_reserved_for_requested_final_score": sorted(
            {f"gen_n{level}_{number:03d}" for level in (1, 2) for number in range(1, 14)}
            - selected_set
        ),
        "local_documents": dict(local_split_docs),
        "local_documents_without_source_annotations_by_split": local_unannotated_by_split,
        "local_negative_context_groups_by_split": negative_context_groups_by_split,
        "local_negative_context_split_policy": "sha256_seed42_with_complete_context_swaps_to_hold_out_negative_groups_in_validation_and_test",
        "local_context_groups": {
            split: len({context_groups[doc.document_id] for doc in documents if doc.source == "local" and doc.split == split})
            for split in ("train", "validation", "test")
        },
        "annotation_transformations": dict(transformations),
        "local_documents_without_source_annotations": sum(not by_doc.get(doc_id) for doc_id in assignments),
        "annotation_rows_retained_for_loss": sum(len(doc.annotations) for doc in documents),
        "annotation_rows": sum(len(doc.annotations) for doc in documents),
        "groups_are_disjoint": True,
        "exact_text_duplicates_crossing_splits": 0,
    }
    return documents, audit


def _hash_selected_files(files: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for doc_id, path in sorted(files.items()):
        digest.update(doc_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _make_document(
    doc_id: str,
    source: str,
    split: str,
    text: str,
    source_sha: str,
    rows: list[dict[str, str]],
    transformations: Counter,
) -> PreparedDocument:
    header_end = _header_boundary(text)
    candidates: list[dict[str, Any]] = []
    ignored: list[tuple[int, int, str]] = []
    confirmed_negatives: list[tuple[int, int, str]] = []
    ignored_annotations: list[dict[str, Any]] = []
    for row in rows:
        start, end = int(row["inicio"]), int(row["fim"])
        if start < 0 or end <= start or end > len(text):
            raise ValueError(f"Span inválido em {doc_id}/{row['citacao_id']}.")
        if text[start:end] != _normalize_annotation_surface(row["trecho"]):
            raise ValueError(f"Span não corresponde ao texto LF em {doc_id}/{row['citacao_id']}.")
        if row["trecho"] != _normalize_annotation_surface(row["trecho"]):
            transformations["escaped_newline_annotation_surface"] += 1
        distractor = _confirmed_header_distractor(text, start, end, header_end)
        if distractor:
            confirmed_negatives.append((start, end, distractor))
            ignored_annotations.append({
                "annotation_id": row["citacao_id"], "start": start, "end": end,
                "decision": "supervised_as_negative", "reason": distractor,
            })
            transformations[distractor] += 1
            continue
        candidates.append(
            {
                "start": start,
                "end": end,
                "kind": "JURISPRUDENCIA" if row["tipo"] == "jurisprudencia" else "LEGISLACAO",
                "annotation_id": row["citacao_id"],
            }
        )

    sorted_rows = sorted(candidates, key=lambda row: (row["start"], row["end"]))
    kept: list[dict[str, Any]] = []
    cluster: list[dict[str, Any]] = []
    cluster_end = -1

    def finish_cluster() -> None:
        if len(cluster) > 1:
            for item in cluster:
                ignored.append((item["start"], item["end"], "ambiguous_overlap"))
                ignored_annotations.append({
                    "annotation_id": item["annotation_id"], "start": item["start"],
                    "end": item["end"], "decision": "ignored_from_loss",
                    "reason": "ambiguous_overlap",
                })
                transformations["ambiguous_overlap"] += 1
        elif cluster:
            kept.append(cluster[0])

    for annotation in sorted_rows:
        if cluster and annotation["start"] >= cluster_end:
            finish_cluster()
            cluster = []
            cluster_end = -1
        if not cluster:
            cluster_end = annotation["end"]
        else:
            cluster_end = max(cluster_end, annotation["end"])
        cluster.append(annotation)
    finish_cluster()
    ignored.sort()
    return PreparedDocument(
        document_id=doc_id,
        source=source,
        split=split,
        level=int(doc_id.split("_n", 1)[1][0]) if "_n" in doc_id else 1,
        text=text,
        annotations=kept,
        ignored_regions=ignored,
        raw_file_sha256=source_sha,
        ignored_annotations=ignored_annotations,
        negative_regions=confirmed_negatives,
    )


def _make_windows(document: PreparedDocument, tokenizer: Any) -> tuple[list[dict[str, Any]], list[int], list[list[int]]]:
    encoding = tokenizer(
        document.text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoding["input_ids"])
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    if not token_ids:
        return [], [], []
    labels = [0] * len(token_ids)
    weights = [1.0] * len(token_ids)

    for start, end, _reason in document.ignored_regions:
        for index, (token_start, token_end) in enumerate(offsets):
            if token_start < end and start < token_end:
                labels[index] = -100
                weights[index] = 0.0

    name_to_id = {name: index for index, name in enumerate(NER_LABELS)}
    for annotation in document.annotations:
        covered = [
            index for index, (token_start, token_end) in enumerate(offsets)
            if token_start < annotation["end"] and annotation["start"] < token_end
        ]
        if not covered:
            raise ValueError(f"Tokenizer não produziu tokens para {document.document_id}/{annotation['annotation_id']}.")
        for place, token_index in enumerate(covered):
            labels[token_index] = name_to_id[("B-" if place == 0 else "I-") + annotation["kind"]]
            weights[token_index] = 1.0

    empty_with_specials = list(tokenizer.build_inputs_with_special_tokens([]))
    special_count = len(empty_with_specials)
    payload_size = WINDOW_SIZE - special_count
    stride = payload_size - WINDOW_OVERLAP
    if payload_size <= 0 or stride <= 0:
        raise ValueError("Tamanho e sobreposição das janelas são incompatíveis.")

    windows = []
    ranges = []
    for start in range(0, len(token_ids), stride):
        end = min(start + payload_size, len(token_ids))
        body = token_ids[start:end]
        model_ids = list(tokenizer.build_inputs_with_special_tokens(body))
        prefix_matches = [
            prefix
            for prefix in range(special_count + 1)
            if model_ids[:prefix] == empty_with_specials[:prefix]
            and model_ids[prefix:prefix + len(body)] == body
            and model_ids[prefix + len(body):] == empty_with_specials[prefix:]
        ]
        if len(prefix_matches) != 1:
            raise ValueError("Não foi possível alinhar especiais do tokenizer com os tokens da janela.")
        prefix = prefix_matches[0]
        positions = list(range(prefix, prefix + len(body)))
        if len(positions) != len(body) or len(model_ids) > WINDOW_SIZE:
            raise ValueError("Tokenizer não preservou o alinhamento da janela com os offsets.")
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(body)
        window_labels = [-100] * len(model_ids)
        window_weights = [0.0] * len(model_ids)
        for offset, position in enumerate(positions):
            global_index = start + offset
            window_labels[position] = labels[global_index]
            window_weights[position] = weights[global_index]
        windows.append(
            {
                "input_ids": model_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": [1] * len(model_ids),
                "labels": window_labels,
                "loss_weights": window_weights,
                "auxiliary_labels": [-100] * len(model_ids),
                "auxiliary_weights": [0.0] * len(model_ids),
                "pseudo_confidences": [0.0] * len(model_ids),
                "body_positions": positions,
                "token_start": start,
                "token_end": end,
            }
        )
        ranges.append((start, end))
        if end == len(token_ids):
            break
    # Each unambiguous source token contributes one unit of citation supervision,
    # regardless of how many overlapping windows contain it.
    appearances = [0] * len(token_ids)
    for start, end in ranges:
        for token_index in range(start, end):
            if labels[token_index] != -100:
                appearances[token_index] += 1
    for window, (start, end) in zip(windows, ranges):
        for local_index, position in enumerate(window["body_positions"]):
            global_index = start + local_index
            if labels[global_index] != -100:
                window["loss_weights"][position] = 1.0 / appearances[global_index]
    return windows, labels, [list(item) for item in offsets]


def _teacher_auxiliary_targets(
    document: PreparedDocument,
    windows: list[dict[str, Any]],
    token_labels: list[int],
    teacher: Any,
    tokenizer: Any,
    torch: Any,
    batch_size: int = 4,
    excluded_regions: list[tuple[int, int, str]] | None = None,
    token_offsets: list[tuple[int, int]] | None = None,
) -> tuple[list[int], list[float]]:
    import numpy as np

    token_count = len(token_labels)
    sums = np.zeros((token_count, len(NER_LABELS)), dtype=np.float32)
    counts = np.zeros(token_count, dtype=np.float32)
    device = next(teacher.parameters()).device
    for offset in range(0, len(windows), batch_size):
        batch = windows[offset:offset + batch_size]
        maximum = max(len(window["input_ids"]) for window in batch)
        pad_id = tokenizer.pad_token_id
        padded_ids = []
        padded_mask = []
        padded_types = []
        for window in batch:
            difference = maximum - len(window["input_ids"])
            if tokenizer.padding_side == "right":
                padded_ids.append(window["input_ids"] + [pad_id] * difference)
                padded_mask.append(window["attention_mask"] + [0] * difference)
                padded_types.append(window["token_type_ids"] + [0] * difference)
            else:
                padded_ids.append([pad_id] * difference + window["input_ids"])
                padded_mask.append([0] * difference + window["attention_mask"])
                padded_types.append([0] * difference + window["token_type_ids"])
        inputs = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long, device=device),
            "token_type_ids": torch.tensor(padded_types, dtype=torch.long, device=device),
        }
        with torch.inference_mode():
            logits = teacher(**inputs).logits.float()
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        for row_index, window in enumerate(batch):
            start = window["token_start"]
            for local_index, model_position in enumerate(window["body_positions"]):
                global_index = start + local_index
                sums[global_index] += probabilities[row_index, model_position]
                counts[global_index] += 1.0
    del teacher
    sums /= np.maximum(counts[:, None], 1.0)
    id_to_name = {index: name for index, name in enumerate(NER_LABELS)}
    auxiliary_ids = {index for index, name in id_to_name.items() if name in AUXILIARY_LABELS}
    confidence_by_token = [0.0] * token_count
    ignored_positions = {
        index for index, label in enumerate(token_labels) if label == -100
    }
    for start, end, _reason in excluded_regions or []:
        ignored_positions.update(
            index for index, (token_start, token_end) in enumerate(token_offsets or [])
            if token_start < end and start < token_end
        )
    for token_index, label in enumerate(token_labels):
        if label != 0 or token_index in ignored_positions:
            continue
        scores = sums[token_index]
        predicted = int(scores.argmax())
        if predicted in auxiliary_ids and float(scores[predicted]) >= AUXILIARY_CONFIDENCE_THRESHOLD:
            token_labels[token_index] = predicted
            confidence_by_token[token_index] = float(scores[predicted])
    id_to_name = {index: name for index, name in enumerate(NER_LABELS)}
    previous_auxiliary_type: str | None = None
    for token_index, label in enumerate(token_labels):
        name = id_to_name.get(label, "O")
        if name.startswith("B-") and name[2:] in {"ORGANIZACAO", "PESSOA", "TEMPO", "LOCAL"}:
            previous_auxiliary_type = name[2:]
        elif name.startswith("I-") and name[2:] in {"ORGANIZACAO", "PESSOA", "TEMPO", "LOCAL"}:
            if previous_auxiliary_type != name[2:]:
                before = name
                token_labels[token_index] = NER_LABELS.index("B-" + name[2:])
                start, end = (token_offsets or [(0, 0)] * token_count)[token_index]
                document.teacher_label_repairs.append({
                    "token_index": token_index, "start": start, "end": end,
                    "from": before, "to": "B-" + name[2:],
                    "reason": "teacher_bio_sequence_repaired",
                })
            previous_auxiliary_type = name[2:]
        else:
            previous_auxiliary_type = None
    return token_labels, confidence_by_token


def _feature_windows(
    document: PreparedDocument,
    tokenizer: Any,
    teacher: Any | None,
    torch: Any,
    teacher_batch_size: int = 4,
) -> list[dict[str, Any]]:
    windows, labels, token_offsets = _make_windows(document, tokenizer)
    if teacher is not None and document.split == "train":
        labels, pseudo_confidences = _teacher_auxiliary_targets(
            document, windows, labels, teacher, tokenizer, torch,
            batch_size=teacher_batch_size,
            excluded_regions=document.negative_regions,
            token_offsets=token_offsets,
        )
        name_to_id = {name: index for index, name in enumerate(NER_LABELS)}
        aux_ids = {name_to_id[name] for name in AUXILIARY_LABELS}
        for window in windows:
            start = window["token_start"]
            for body_index, model_position in enumerate(window["body_positions"]):
                global_index = start + body_index
                label = labels[global_index]
                if label in aux_ids and window["labels"][model_position] == 0:
                    window["auxiliary_labels"][model_position] = label
                    window["auxiliary_weights"][model_position] = window["loss_weights"][model_position]
                    window["pseudo_confidences"][model_position] = pseudo_confidences[global_index]

    for window_index, window in enumerate(windows):
        window.pop("body_positions")
        window_token_start = window.pop("token_start")
        window_token_end = window.pop("token_end")
        window["document_id"] = document.document_id
        window["source"] = document.source
        window["level"] = document.level
        window["sampling_group"] = _document_sampling_group(document.document_id, document.source)
        window["base_document_id"] = document.base_document_id or document.document_id
        window["augmentation_rate"] = document.augmentation_rate
        window["window_index"] = window_index
        window["window_token_start"] = window_token_start
        window["window_token_end"] = window_token_end
    return windows


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _write_gold(path: Path, docs: list[PreparedDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["nivel", "documento_id", "citacao_id", "inicio", "fim", "trecho", "tipo", "classificacao", "id_canonico"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for document in sorted(docs, key=lambda item: item.document_id):
            allowed_ids = {annotation["annotation_id"] for annotation in document.annotations}
            original_rows = _read_annotations_for_document(document)
            for row in original_rows:
                try:
                    if row["citacao_id"] not in allowed_ids:
                        continue
                    start, end = int(row["inicio"]), int(row["fim"])
                    if not (0 <= start < end <= len(document.text)):
                        continue
                    reason = _confirmed_header_distractor(document.text, start, end, _header_boundary(document.text))
                    if reason:
                        continue
                    if document.text[start:end] != _normalize_annotation_surface(row["trecho"]):
                        continue
                except (KeyError, ValueError):
                    continue
                writer.writerow({**row, "trecho": document.text[start:end]})


def _write_original_gold(path: Path, docs: list[PreparedDocument]) -> None:
    """Retain the source annotations, including rows adapted out of the loss."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["nivel", "documento_id", "citacao_id", "inicio", "fim", "trecho", "tipo", "classificacao", "id_canonico"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for document in sorted(docs, key=lambda item: item.document_id):
            for row in _read_annotations_for_document(document):
                writer.writerow({field: row.get(field, "") for field in fields})


_ORIGINAL_ROWS: dict[str, list[dict[str, str]]] = {}


def _read_annotations_for_document(document: PreparedDocument) -> list[dict[str, str]]:
    return _ORIGINAL_ROWS.get(document.document_id, [])


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
            count += 1
    return count


def prepare_campaign(
    local_text_dir: Path,
    local_annotation_path: Path,
    official_text_dir: Path,
    official_annotation_path: Path,
    source_model_dir: Path,
    output_dir: Path,
    tokenizer: Any,
    teacher: Any | None,
    torch: Any,
    teacher_batch_size: int = 4,
) -> dict[str, Any]:
    """Create train/dev/test JSONL and LF text copies under the run directory."""
    global _ORIGINAL_ROWS
    local_rows = _read_annotations(local_annotation_path)
    official_rows = _read_annotations(official_annotation_path, set(_official_selection()))
    _ORIGINAL_ROWS = defaultdict(list)
    for row in local_rows:
        _ORIGINAL_ROWS[row["documento_id"]].append(row)
    selected_set = set(_official_selection())
    for row in official_rows:
        if row["documento_id"] in selected_set:
            _ORIGINAL_ROWS[row["documento_id"]].append(row)

    documents, audit = _collect_documents(
        local_text_dir, local_annotation_path, official_text_dir, official_annotation_path
    )
    prepared = output_dir / "prepared"
    all_features: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    split_docs: dict[str, list[PreparedDocument]] = {"train": [], "validation": [], "test": []}
    auxiliary_token_counts = Counter()
    windows_by_source = Counter()
    for document in documents:
        split_docs[document.split].append(document)
        _write_text(prepared / document.source / document.split / "txt" / f"{document.document_id}.txt", document.text)
        features = _feature_windows(
            document, tokenizer, teacher if document.split == "train" else None,
            torch, teacher_batch_size=teacher_batch_size,
        )
        auxiliary_ids = {
            index for index, label in enumerate(NER_LABELS)
            if label in AUXILIARY_LABELS
        }
        pseudo_tokens = sum(
            weight > 0.0 and label in auxiliary_ids
            for feature in features
            for label, weight in zip(feature["auxiliary_labels"], feature["auxiliary_weights"])
        )
        auxiliary_token_counts[f"{document.source}_before_official_repeat"] += pseudo_tokens
        windows_by_source[f"{document.source}_before_official_repeat"] += len(features)
        if document.source == "official" and document.split == "train":
            features = features * OFFICIAL_REPEAT
            auxiliary_token_counts["official_repeat_exposure_extra"] += pseudo_tokens * (OFFICIAL_REPEAT - 1)
            windows_by_source["official_repeat_exposure_extra"] += len(features) - len(features) // OFFICIAL_REPEAT
        windows_by_source[f"{document.source}_sampled"] += len(features)
        all_features[document.split].extend(features)

    feature_counts = {}
    for split, rows in all_features.items():
        train_name = "train.jsonl" if split == "train" else f"local_{split}.jsonl"
        feature_counts[split] = _write_jsonl(prepared / train_name, rows)
    _write_gold(prepared / "local_validation.csv", split_docs["validation"])
    _write_gold(prepared / "local_test.csv", split_docs["test"])
    _write_original_gold(prepared / "local_validation_original.csv", split_docs["validation"])
    _write_original_gold(prepared / "local_test_original.csv", split_docs["test"])

    per_doc = [
        {
            "document_id": doc.document_id,
            "source": doc.source,
            "split": doc.split,
            "level": doc.level,
            "text_sha256": hashlib.sha256(doc.text.encode("utf-8")).hexdigest(),
            "source_file_sha256": doc.raw_file_sha256,
            "annotation_count_after_filter": len(doc.annotations),
            "ignored_regions": [
                {"start": start, "end": end, "reason": reason}
                for start, end, reason in doc.ignored_regions
            ],
            "confirmed_negative_regions": [
                {"start": start, "end": end, "reason": reason}
                for start, end, reason in doc.negative_regions
            ],
            "ignored_annotation_decisions": doc.ignored_annotations,
            "teacher_label_repairs": doc.teacher_label_repairs,
        }
        for doc in documents
    ]
    audit.update(
        {
            "window_size_total_tokens": WINDOW_SIZE,
            "window_overlap_tokens": WINDOW_OVERLAP,
            "official_window_repeat": OFFICIAL_REPEAT,
            "tokenizer_vocab_size": len(tokenizer),
            "tokenizer_is_fast": bool(tokenizer.is_fast),
            "tokenizer_max_length": tokenizer.model_max_length,
            "features_by_split": feature_counts,
            "windows_by_source_and_sampling": dict(windows_by_source),
            "auxiliary_pseudolabel_tokens_by_source_sampled": dict(auxiliary_token_counts),
            "source_checkpoint_tree_sha256": sha256_tree(source_model_dir),
            "source_checkpoint_model_sha256": sha256_file(source_model_dir / "model.safetensors"),
            "prepared_artifacts_sha256": {
                path.relative_to(prepared).as_posix(): sha256_file(path)
                for path in sorted(prepared.rglob("*"))
                if path.is_file() and path.name != "dataset_manifest.json"
            },
            "documents": per_doc,
        }
    )
    write_json(prepared / "dataset_manifest.json", audit)
    return audit
