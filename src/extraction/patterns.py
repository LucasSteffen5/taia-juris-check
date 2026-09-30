# -*- coding: utf-8 -*-
"""Padrões regex compilados para extração de jurisprudência e leis."""
import re

SP = r"[\s\xa0\n]+"
SPOPT = r"[\s\xa0\n]*"

# Siglas de Unidades Federativas reais do Brasil
UFS_LIST = (
    r"(?:AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)"
)

# Classes processuais
CLASSES_REGEX = (
    r"(?:"
    r"s[uú]mula(?:" + SP + r"vinculante)?|5[uú]mula|s[uú]m\."
    r"|(?:terceiro" + SP + r")?ag(?:\.|\s*)reg(?:\.|\s*|imental)?"
    r"|(?:embargos" + SP + r"de" + SP + r"declara[çc][ãa]o|edcl|ed|eds)"
    r"|(?:agravo(?:" + SP + r"(?:interno|regimental|em" + SP + r"recurso" + SP + r"especial|de" + SP + r"instrumento))?)"
    r"|(?:recurso" + SP + r"(?:especial(?:" + SP + r"eleitoral)?|extraordin[áa]rio|em" + SP + r"habeas" + SP + r"corpus|em" + SP + r"mandado" + SP + r"de" + SP + r"seguran[çc]a|em" + SP + r"sentido" + SP + r"estrito|ordin[áa]rio))"
    r"|(?:reclama[çc][ãa]o|rcl|recl\.)"
    r"|(?:habeas" + SP + r"corpus|rhc|r\.hc|hc|h\.c\.)"
    r"|(?:suspens[ãa]o" + SP + r"de" + SP + r"liminar(?:" + SP + r"e" + SP + r"de" + SP + r"senten[çc]a)?|sl|sls)"
    r"|(?:apela[çc][ãa]o|apl|rse|rms|respe|arespel|aresp|resp|rec\.\s*esp\.|r\.esp\.|agresp|agint|agrg|agr|r\.?e\.?|ar|arr|rr|airr|r-rp)"
    r"|tem[aã](?:" + SP + r"da" + SP + r"repercuss[ãa]o" + SP + r"geral)?"
    r")"
)

CONNECTIVE = r"(?:" + SP + r"(?:no|na|nos|nas|em|de|do|da|dos|das)" + SP + CLASSES_REGEX + r")*"
NUM_PREFIX = r"(?:" + SP + r"(?:n[ºo°\.]|n\.\s*|n\s+|no\s+|n\xba\s*))?"
UF_REGEX = r"(?:[\s\xa0/\-–\n]+" + SPOPT + r"(?:" + UFS_LIST + r"\b|\(" + SPOPT + UFS_LIST + SPOPT + r"\)))"

# Números: exige ao menos um dígito (\d+) em cada bloco para evitar palavras em português
NUM_TOKEN = r"[\d\.\xa0\-_lIOoGgSs]*\d+[\d\.\xa0\-_lIOoGgSs]*"
NUM_BODY = r"\s*" + NUM_TOKEN + r"(?:\s+" + NUM_TOKEN + r")*"

# 1. Padrão Processual Clássico
RE_JURIS_CLASSICA = re.compile(
    r"\b("
    + CLASSES_REGEX
    + CONNECTIVE
    + NUM_PREFIX
    + NUM_BODY
    + r"(?:" + UF_REGEX + r")?"
    + r"(?:" + SP + r"da" + SP + r"repercuss[ãa]o" + SP + r"geral)?"
    + r")",
    flags=re.IGNORECASE,
)

# 2. Padrão CNJ
RE_JURIS_CNJ = re.compile(
    r"\b("
    r"(?:(?:processo" + SP + r"(?:n[ºo°\.]\s*)?)?TST-[\w\-\s\xa0\.\n]+\d+[\d\-\.\s\xa0\n]{10,35}"
    r"|(?:[A-Z]{1,4}-[A-Z]{1,4}|[A-Z]{2,6}|AgR-[A-Z]{2,6}|EDs?" + SP + r"no" + SP + r"[A-Z\-]+|Ag\.\s*Int\.\s*No|AgInt\s+No|APL|RSE|REspe\.?)"
    + NUM_PREFIX
    + r"[\s\xa0\n]*\d{1,7}[\s\xa0\-\.\n]*\d{2}[\s\xa0\-\.\n]*\d{4}[\s\xa0\-\.\n]*\d[\s\xa0\-\.\n]*\d{2}[\s\xa0\-\.\n]*\d{4}"
    r"(?:" + UF_REGEX + r")?"
    r"|"
    r"(?:[A-Z]{1,4}-[A-Z]{1,4}|AgR-[A-Z]+|ED" + SP + r"no" + SP + r"[A-Z\-]+|Ag\.\s*Int\.\s*No|AgInt\s+No|APL|RSE|REspe\.?)"
    + NUM_PREFIX
    + r"[\s\xa0\n]*\d{14,24}"
    r"(?:" + UF_REGEX + r")?"
    r")"
    r")",
    flags=re.IGNORECASE,
)

# 3. Padrão Súmula
RE_SUMULA = re.compile(
    r"\b("
    r"(?:S[uú]mula" + SP + r"Vinculante|5[uú]mula" + SP + r"Vinculante|S[uú]mula|5[uú]mula|S[uú]m\.)"
    + NUM_PREFIX
    + r"\s*\d+"
    r"(?:" + SP + r"(?:do|da|de)" + SP + r"(?:STF|STJ|TST|TSE|STM))?"
    r")",
    flags=re.IGNORECASE,
)

# 4. Padrão Lei
SUBDIV = (
    r"(?:," + SPOPT + r"(?:[IVXLCDM]+|§" + SPOPT + r"\d+[ºo°]?(?:-[A-Za-z])?|inciso" + SP + r"[IVXLCDM]+|par[áa]grafo" + SP + r"[uú]nico|'?[a-z]'?|[A-Za-z]\b))*"
    + SPOPT + r",?"
)

RE_LEI = re.compile(
    r"\b("
    r"(?:art(?:igo)?\.?" + SP + r"[\d\.]+[ºo°]?"
    + SUBDIV
    + SP + r"(?:do|da|de|dos|das|no|na)" + SP +
    r"(?:"
    r"Constitui[çc][ãa]o(?:" + SP + r"(?:Federal|Fedcral|da" + SP + r"Rep[úu]blica|local|de" + SP + r"\d{4}))?"
    r"|C[óo]digo(?:" + SP + r"(?:de" + SP + r"Processo" + SP + r"Civil|Civil|de" + SP + r"Defesa" + SP + r"do" + SP + r"Consumidor|Penal(?:" + SP + r"Militar)?|de" + SP + r"Processo" + SP + r"Penal|Eleitoral))?"
    r"|Consolida[çc][ãa]o" + SP + r"das" + SP + r"Leis" + SP + r"do" + SP + r"Trabalho"
    r"|(?:Constitui[çc][ãa]o(?:\s+Federal|\s+Fedcral)?|CF)(?:\s*/\s*\d{2,4})"
    r"|CPC|CC|CLT|CPP|CPM|CDC"
    r"|Lei(?:" + SP + r"Complementar)?(?:" + SP + r"n[ºo°\.]\s*|" + SP + r"n\.\s*|" + SP + r")[\d\.]+(?:/\d{4})?"
    r")"
    r")"
    r")",
    flags=re.IGNORECASE,
)

# 5. Padrão Citações Vagas (Incompletas por definição):
RE_VAGA = re.compile(
    r"\b("
    r"(?:julgado|ac[óo]rd[ãa]o|precedente|recurso" + SP + r"em" + SP + r"habeas" + SP + r"corpus|reclama[çc][ãa]o|rcl|apl|agravo" + SP + r"em" + SP + r"recurso" + SP + r"especial)"
    r"(?:" + SP + r"(?:do|da|de)" + SP + r"(?:STF|STJ|TST|TSE|STM))?"
    r"(?:," + SPOPT + "|" + SP + ")?"
    r"(?:julgado" + SP + r"em" + SP + r"\d{4}|prof[ec]rido" + SP + r"em" + SP + r"\d{4}|de" + SP + r"\d{4})"
    r"(?:," + SPOPT + "|" + SP + ")?"
    r"(?:sob" + SP + r"relatoria" + SP + r"d[ec]|pela" + SP + r"relatoria" + SP + r"d[ec]|da" + SP + r"relatoria" + SP + r"d[ec]|rel\.\s*min\.|relatora" + SP + r"d[ec])"
    + SP + r"[A-Za-zÀ-ÿ\s\xa0\.\n]{3,45}"
    r")",
    flags=re.IGNORECASE,
)

# Generic legal allusions are citable but cannot identify one canonical record.
RE_VAGA_GENERIC = re.compile(
    r"\b(?:conforme|segundo|à\s+luz\s+da|nos\s+termos\s+da)\s+"
    r"(?:(?:a|o)\s+)?(?:jurisprud[êe]ncia|entendimento|orienta[çc][ãa]o)"
    r"(?:\s+(?:pac[íi]fica|consolidada|dominante|majorit[áa]ria|sumulada))?"
    r"(?:\s+(?:do|da|de)\s+(?:tribunal|corte|STF|STJ|TST|TSE|STM))?",
    flags=re.IGNORECASE,
)
