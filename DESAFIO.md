# Overview

LLMs generativos já são usados para redigir pareceres, petições e memorandos jurídicos. Um dos riscos mais documentados desses sistemas é a **alucinação de fontes**: o modelo cita jurisprudência ou dispositivos legais que não existem, ou apresenta citações tão vagas que não podem ser verificadas. Nos últimos anos, tribunais no Brasil e no exterior já sancionaram advogados por protocolar peças com citações inventadas geradas por IA.

Neste desafio, você assume o papel de quem precisa construir o verificador automático que falta nesse fluxo de trabalho: um sistema que lê um documento judicial, identifica todas as citações de jurisprudência e de lei, e decide — para cada uma — se ela é **real**, **inventada** ou **incompleta**.

## A tarefa

Seu sistema recebe documentos judiciais em texto (`.txt`) e deve, para cada documento, identificar todas as citações e classificar cada uma em três classes:

- **`real`** — a citação resolve a um único registro da base canônica do desafio: tem identificadores suficientes e a busca confirma sua existência. Exige entregar o `id_canonico` correto — acertar o rótulo sem o link não conta.
- **`inventada`** — identificadores suficientes para buscar, mas nenhum registro correspondente existe na cobertura congelada.
- **`incompleta`** — informação insuficiente para formular a consulta ("conforme jurisprudência pacífica do tribunal"), ou suficiente para buscar mas insuficiente para identificar um único registro.

A distinção entre `inventada` e `incompleta` importa em produção: a primeira é alucinação ativa (o LLM inventou um número de acórdão); a segunda é evasiva. Sistemas reais tratam cada caso de forma diferente — incompletas vão para revisão humana; inventadas devem ser bloqueadas.

O conjunto tem **dois níveis de dificuldade**: Nível 1 (formato padrão, peso 1×) e Nível 2 (ruído de OCR e variação de superfície, peso 2×). Os detalhes de dados, base canônica e gabarito estão na aba **Data**.

## Como participar

1. **Monte sua equipe**: participação individual ou em equipes de até 4 pessoas (mesclagem de equipes pelo próprio Kaggle), apenas estudantes, do Brasil — a elegibilidade é verificada pela organização (ver Rules).
2. **Receba os dados**: documentos de entrada (`txt/`), base canônica SQLite e gabarito da amostra de desenvolvimento estão na aba **Data**, junto com o conversor de submissão e o script da métrica.
3. **Desenvolva sua solução** — apenas modelos e ferramentas de pesos e código **abertos** (veja Rules). Qualquer dataset público pode ser usado no treino.
4. **Submeta**: sua solução produz um JSON por documento (contrato na aba Data); o conversor `json_to_submission.py` gera o `submission.csv` que você envia aqui. A métrica oficial roda no servidor e o leaderboard atualiza na hora.
5. **Publique seu código**: as soluções do topo do ranking passam por verificação de reprodutibilidade ao final — o repositório e o commit que produziram as saídas serão coletados junto aos finalistas.
## Avaliação

A avaliação tem **duas fases**:

1. **Fase de treino (agora)** — o material distribuído é a amostra de desenvolvimento, com gabarito aberto. O leaderboard roda sobre ela e é **referencial**: serve para validar seu pipeline de ponta a ponta, não define o resultado.
2. **Fase de avaliação final** — o conjunto final, cego, está em construção pela organização. Quando for ativado, o leaderboard **reinicia**: passa a usar **40% dele** (parte pública do leaderboard), e o ranking final é calculado sobre os **60% restantes**, mantidos em sigilo até o encerramento — otimizar demais para o leaderboard não garante nada no resultado final. Submissões da fase de treino não contam para o ranking final.

O alinhamento entre suas citações e o gabarito é por **sobreposição de spans com IoU ≥ 0,5** (matching 1-para-1 pelo maior IoU). Citação do gabarito sem par vira erro de recall; predição sem par vira falso positivo. Sobre os pares casados, a nota de cada nível é montada em três passos:

**Passo 1 — macro-F1 das 3 classes.** F1 por classe (`real`, `inventada`, `incompleta`), com média simples entre elas — assim a classe `inventada`, rara mas central ao desafio, pesa tanto quanto a `real`. Para a classe `real`, só conta acerto se o `id_canonico` entregue for o doc_id esperado no gabarito.

**Passo 2 — penalidade do erro grave.** O pior erro é carimbar uma alucinação como verificada:

```
τ = fração das citações `inventada` do gabarito preditas como `real`
s = macroF1 · (1 − 0,5 · τ)
```

Se todas as alucinações "vazarem" como real, a nota cai pela metade.

**Passo 3 — bônus de calibração (até 10%).** O campo `confianca` é opcional; se enviado, medimos o Brier score sobre os pares casados e aplicamos `score = s · (1 + 0,10 · (1 − brier))`. Confiança alta quando acerta e baixa quando erra aproxima o bônus de +10%. Quem não envia confiança não ganha nem perde.

**Combinação dos níveis:**

```
score_final = (1 · score_Nível1 + 2 · score_Nível2) / 3
```

Uma submissão perfeita com `confianca = 1.0` pontua **1.1000**.

## Formato de submissão

Sua solução produz um JSON por documento (contrato schema 1.2, na aba Data); o conversor `json_to_submission.py` (incluído no material) gera o `submission.csv` — **uma linha por documento**:

```
documento_id,citacoes
gen_n1_001,"469,498,incompleta,-,0.9|589,652,incompleta,-,0.8"
gen_n1_002,-
```

Cada citação é `inicio,fim,classe,id_canonico,confianca` (campos ausentes viram `-`), com citações separadas por `|`. Documento **sem** citações leva `-` na célula — nunca deixe a célula vazia, e todo `documento_id` do conjunto precisa estar presente.

**Reproduza localmente:** o script de avaliação (`kaggle_metric.py`, na aba Data) é o mesmo que roda aqui — você reproduz na sua máquina exatamente o score do leaderboard, nível a nível, incluindo o matching.
## FAQ


**Posso usar APIs comerciais, como modelos de linguagem pagos?** Não. Este desafio aceita **apenas modelos e ferramentas de pesos e código abertos** — a organização precisa conseguir executar sua solução de ponta a ponta, sem chaves de API nem serviços pagos. Servir um modelo de pesos abertos via API paga durante o desenvolvimento é permitido, desde que o mesmo modelo e revisão sejam executáveis offline pela organização. Qualquer dataset **público** pode ser usado no treino.

**Como funciona o leaderboard? Meu score final pode mudar?** Em duas fases. Agora, na fase de treino, o leaderboard pontua contra a amostra de desenvolvimento (gabarito aberto) e é referencial — use-o para validar seu pipeline. Quando o conjunto de avaliação final for ativado, o leaderboard reinicia e passa a usar a parte pública (40%) dele; o ranking final vem dos 60% privados, mantidos em sigilo até o encerramento — a posição pública é um bom sinal, mas o resultado final pode mudar.

**Quantas submissões posso fazer?** Múltiplas, com teto diário por equipe (configurado nesta competição).

**O offset (`inicio`/`fim`) pontua?** Não por si só — ele é a chave de junção com o gabarito (IoU ≥ 0,5). Acima do limiar, a borda exata é irrelevante; abaixo, a citação esperada fica sem par (erro de recall) e a predição órfã vira falso positivo. Cuidado com a contagem: codepoints Unicode sobre o texto exatamente como distribuído.

**O campo `confianca` é obrigatório?**Sim: confiança bem calibrada (Brier baixo) rende bônus de até 10% no score.

**Em que língua são os documentos?** Português brasileiro, com citações de jurisprudência e legislação brasileiras.

## Dataset Description

### Arquivos

| **Arquivo** | **O que é** |
| --- | --- |
| `txt/` | Os 26 documentos de entrada (`.txt`, UTF-8) |
| `desafio1_bracis.db` | A base canônica SQLite (aprox. 94 MB, 1.014 registros) |
| `goldenset_offsets.csv` | O gabarito da amostra de desenvolvimento (192 citações) |
| `json_to_submission.py` | Conversor: JSONs do contrato → `submission.csv` |
| `kaggle_metric.py` | O script de avaliação — o mesmo que roda neste leaderboard |

## A tarefa

**Encontrar as citações num documento judicial e classificar cada uma em três classes.**

A entrada é um `.txt` por documento. A saída é um JSON por documento, com uma entrada por citação: o span em codepoints Unicode (`inicio`, `fim` exclusivo), o trecho literal, o tipo (`lei` ou `jurisprudencia`) e a classe. Opcionalmente, a `confianca` (0–1) de que a classe — e o link, se `real` — está correta: ela alimenta o bônus de calibração de até 10%, e quem não a envia não ganha nem perde por isso.

> **Submissão no Kaggle:** sua solução produz os JSONs e o conversor `json_to_submission.py` (incluído aqui na aba Data) gera o `submission.csv` que você envia — uma linha por documento; documento sem citações leva `-` na célula, nunca vazia. A métrica que roda no servidor é a cópia exata de `kaggle_metric.py`, também nesta aba.

O alinhamento entre a predição e o gabarito é por sobreposição de spans com **IoU ≥ 0,5** — não é preciso acertar a borda exata, mas quem não entrega o span de uma citação não consegue classificá-la, e isso conta como erro de recall.

| **`real`** | **`inventada`** | **`incompleta`** |
| --- | --- | --- |
| A citação resolve a um único **registro** da base canônica. Exige `id_canonico`: acertar o rótulo sem o doc\_id correto não conta. | Identificadores suficientes para buscar, mas nenhum registro correspondente existe na cobertura congelada. | Informação insuficiente para formular a consulta, ou suficiente para buscar mas insuficiente para identificar um único registro. |

## Os documentos de entrada

**Vinte e seis peças jurídicas sintéticas, divididas em dois níveis de dificuldade — pesos 1× e 2× na nota.**

Esta é a **amostra de treino/desenvolvimento**, distribuída com gabarito para as equipes construírem e depurarem suas soluções — é sobre ela que o leaderboard roda na fase atual (referencial). O **conjunto de avaliação final** — cego — está **em construção** pela organização: terá o mesmo formato, os mesmos níveis e distribuição de classes equivalente, **sem gabarito distribuído**. Quando for ativado, os `.txt` dele serão publicados nesta aba, as saídas do seu sistema sobre eles passam a ser o que você submete, e é dele que saem o leaderboard (40% público) e o ranking final (60% privado).

São petições, pareceres, decisões monocráticas e memoriais fictícios que citam acórdãos reais do acervo. Cada documento tem uma matéria coerente — cível, penal, trabalhista, eleitoral ou militar — que amarra o cabeçalho, os tribunais citados e as normas invocadas: um agravo cível não cita apelação da Justiça Militar.

A nomenclatura dos arquivos já entrega o nível: `gen_n1_001` … `gen_n1_013` são os documentos do nível 1 e `gen_n2_001` … `gen_n2_013`, os do nível 2. O nome do arquivo em `txt/` (sem a extensão) é o `documento_id` usado no gabarito e na submissão.

|  | **Nível 1 — Formato padrão** | **Nível 2 — Ruído e variação** |
| --- | --- | --- |
| **Peso na nota** | 1× | 2× |
| **Documentos** | 13 | 13 |
| **Citações** | 99 | 93 |
| **Tamanho médio** | 3.393 chars | 3.277 chars |
| **Classes** | real 52 · inventada 32 · incompleta 15 | real 44 · inventada 32 · incompleta 17 |
| **O que testa** | Reconhecer a citação e resolver o doc\_id na base canônica. | Normalização robusta antes de verificar: casar variantes de superfície ao mesmo identificador. |

### A mesma classe, escrita de dois jeitos

As citações do nível 2 apontam para registros igualmente válidos — o que muda é a superfície. Todas as amostras abaixo são `real`:

| **Nível 1** | **Nível 2** |
| --- | --- |
| `AREsp nº 1.996.496/RJ` | `AgRg no Rec. Esp. n. 1.522.200 (SC)` |
| `Recurso em Habeas Corpus nº 57.763/PR` | `Recurso em Habeas Corpus nº 93967 - SC` |
| `RSE nº 7000592-58.2025.7.00.0000/DF` | `Rec. Esp. No 1.880.529 ⏎ - SP` |
| `Súmula Vinculante 10` | `5úmula 211 do STJ` |

O ruído do nível 2 combina variantes de abreviação (`REsp` / `R.Esp.` / `Recurso Especial`), formatação do número (`1.741.784` / `1741784` / `1.741. 784`), separador de UF (`/PR`, `- PR`, `(PR)`), confusões de OCR (`0↔O`, `1↔l`, `5↔S`, `m↔rn`) e quebras de linha no meio do identificador.

> **Garantia do ruído:** um dígito nunca é trocado por outro dígito. Isso mudaria a identidade da citação e transformaria uma `real` ruidosa numa `inventada` de fato. Todo ruído aplicado a uma citação real é recuperável por normalização — que é exatamente o que o nível 2 mede.

## Distratores

Os cabeçalhos trazem números que *parecem* citação e não são: número dos autos em formato CNJ, protocolo, inscrição na OAB, `fls. 234/567`, valor da causa. Nenhum está no gabarito — extraí-los conta como falso positivo.

Atenção à distinção no formato CNJ: o número dos autos *do próprio documento*, no cabeçalho, é distrator; a referência a *outro processo* em formato CNJ, no corpo do texto, é citação (como o `RSE nº 7000592-58.2025.7.00.0000/DF` do quadro acima).

## A base canônica

**Um SQLite de aproximadamente 94 MB com os 1.014 registros que definem o universo do desafio. É contra ele que uma citação é real ou inventada.**

O banco é a **cobertura congelada**: um snapshot. Não se consulta base viva, e é isso que torna a avaliação reproduzível. Se um acórdão existe no mundo mas não está aqui, para efeito do desafio ele não existe — e, por construção, isso nunca prejudica ninguém: toda citação `real` dos documentos resolve dentro da cobertura. O caso-limite "registro real fora da cobertura" não ocorre neste dataset; se ocorresse, a citação seria excluída do scoring, não anotada como `inventada`.

| **Coluna** | **Tipo** | **Descrição** |
| --- | --- | --- |
| `documento_id` | TEXT | Chave primária e nome do .txt. `doc_0201` para acórdãos; o próprio doc\_id para súmulas e leis. |
| `id` | INTEGER | **O doc\_id canônico do Jusbrasil.** É este número que vai em `resolucao.id_canonico`. |
| `tribunal` | TEXT | STF, STJ, TSE, TST, STM. Nulo para dispositivos de lei. |
| `ano`, `relator` | INT, TEXT | Só os acórdãos têm. |
| `natureza` | TEXT | acordao · sumula · dispositivo |
| `tipo` | TEXT | jurisprudencia · lei — o enum do contrato. |
| `texto` | TEXT | Inteiro teor. Idêntico a `txt/<documento_id>.txt`. |
| `texto_len` | INTEGER | Comprimento em caracteres. |

`natureza` existe porque `tipo` sozinho não separa acórdão de súmula — os dois são `jurisprudencia`.

| **Natureza** | **Registros** | **O que são** |
| --- | --- | --- |
| `acordao` | 996 | Acórdãos de STF, STJ, TSE, TST e STM. |
| `sumula` | 5 | Súmulas do STJ, STF e TST, incluindo vinculante. |
| `dispositivo` | 13 | Artigos de CPC, CC, CLT, CF/88, CPP, CPM, CDC, Código Eleitoral e LC 64/1990. |

Há ainda uma tabela virtual `documentos_fts` (FTS5, *external content*) indexando o texto integral dos 1.014 registros. Ela não duplica o conteúdo — aponta para `documentos` pelo `rowid`.

## O gabarito

**O `goldenset_offsets.csv` tem uma linha por citação esperada — 192 no total, nos 26 documentos.**

É o gabarito da amostra de desenvolvimento. O gabarito do conjunto final tem o mesmo formato, permanece com a organização e é usado exclusivamente na avaliação oficial.

| **Coluna** | **Descrição** |
| --- | --- |
| `nivel` | 1 ou 2 — define o peso na nota final. |
| `documento_id`, `citacao_id` | Identificam a citação. `gen_n1_004` + `g3` — o `n1`/`n2` no nome indica o nível do documento. |
| `inicio`, `fim` | Span em codepoints Unicode, fim exclusivo. Chave de junção com a predição. |
| `trecho` | Cópia literal de `texto[inicio:fim]`. Quebras de linha aparecem escapadas como `\n`. |
| `tipo` | `lei` ou `jurisprudencia`. |
| `classificacao` | A classe esperada. É o que se pontua. |
| `id_canonico` | O doc\_id do registro que resolve a citação. Só nas `real`. |

## Como usar a base

**O fluxo tem três passos, e a maior parte da dificuldade está no primeiro.**

### 1 · Normalizar o identificador antes de consultar

O índice FTS usa o tokenizador `unicode61`, que quebra em qualquer caractere não alfanumérico. `1.741.784` vira três tokens — `1`, `741`, `784`. A consequência prática:

```sql
-- funciona: a busca por frase reproduz a sequência de tokens
SELECT d.documento_id, d.id, d.tribunal
  FROM documentos_fts JOIN documentos d ON d.rowid = documentos_fts.rowid
 WHERE documentos_fts MATCH '"1.741.784"';   -- 1 resultado

-- não funciona: sem os separadores, é um token só, que não existe no índice
 WHERE documentos_fts MATCH '1741784';        -- 0 resultados
```

Como o nível 2 entrega números sem pontuação com frequência, o pipeline precisa reconstruir a forma canônica: extrair só os dígitos e reagrupá-los de três em três a partir da direita. `"1 741 784"` como frase também casa, porque produz a mesma sequência de tokens.

Números no padrão CNJ — `0600216-46.2020.6.14.0022` — funcionam do mesmo jeito, com busca por frase.

### 2 · Separar o registro de quem apenas o cita

Esta é a armadilha que mais custa precisão. O FTS devolve *qualquer* documento que mencione o número, e acórdãos citam uns aos outros o tempo todo. Uma busca por `"1.276.977"` devolve seis documentos do STF — e **nenhum deles é o RE 1.276.977**. Todos apenas o citam.

O sinal que separa os dois casos é a posição. No documento que *é* o processo, o número está no cabeçalho, nas primeiras dezenas de caracteres. Nos que apenas citam, ele aparece no corpo:

| **Documento** | **Posição** | **Contexto** |
| --- | --- | --- |
| `doc_0201` | 20 | `RECURSO ESPECIAL Nº 1.741.784 - PR …` — é o processo |
| `doc_0001` | 48 | `AG.REG. NA RECLAMAÇÃO 76.532 RIO DE JANEIRO …` — é o processo |
| `doc_0149` | 490 | menciona o RE 1.276.977 na ementa — apenas cita |
| `doc_0001` | 957 | também cita o RE 1.276.977, mais adiante |

Uma abordagem mais robusta que o limiar de posição é fazer o parsing do cabeçalho: classe processual, número e UF vêm sempre nas primeiras linhas, num formato estável por tribunal. Vale construir esse índice uma vez, offline, e consultar por chave em vez de varrer o FTS a cada citação.

### 3 · Contar os candidatos e decidir a classe

Resolvida a consulta, a cardinalidade do resultado dá a classe quase diretamente. Cada citação `real` do desafio resolve para exatamente um registro da base — as duplicatas remanescentes do acervo não têm nenhuma citação apontando para elas.

| **Candidatos encontrados** | **Classe** | **Saída** |
| --- | --- | --- |
| exatamente 1 | `real` | `resolucao.id_canonico` = a coluna `id` do registro |
| 0 | `inventada` | `resolucao: null` |
| 2 ou mais candidatos distintos, sem critério de desempate | `incompleta` | `resolucao: null` |

Uma citação sem identificadores suficientes para sequer formular a consulta — "jurisprudência pacífica desta Corte", "o dispositivo legal de regência" — é `incompleta` sem passar pelo banco. E uma referência do tipo "acórdão do STF de 2024, relatado pelo Ministro Fulano" é buscável, mas casa com dezenas de candidatos distintos: também `incompleta`, por falta de critério de desempate.

```sql
-- a consulta por relator+ano que caracteriza a incompleta "buscável"
SELECT COUNT(*) FROM documentos
 WHERE natureza = 'acordao'
   AND tribunal = 'STF' AND ano = 2024
   AND relator LIKE '%Dino%';   -- dezenas de candidatos
```

### Leis e súmulas resolvem por registro próprio

Buscar "Súmula 83 do STJ" no texto dos acórdãos devolve dezenas de documentos que a mencionam — nenhum deles é a súmula. Os 18 registros de natureza `sumula` e `dispositivo` existem para isso: são o alvo da resolução, não o texto que cita.

```pgsql
-- todos os registros que não são acórdão, com o texto normativo
SELECT documento_id, id, natureza, tipo, substr(texto, 1, 80)
  FROM documentos
 WHERE natureza IN ('sumula', 'dispositivo')
 ORDER BY natureza, id;
```

Como são poucos, o caminho prático é carregá-los em memória e casar por texto normalizado do dispositivo — `art. 373, I, do CPC` e `artigo 373, inciso I, do Código de Processo Civil` apontam para o mesmo `id`.

> **Não confunda as duas colunas de ID:** `documento_id` é a chave interna do acervo e o nome do arquivo. `id` é o doc\_id do Jusbrasil, e é ele que vai em `resolucao.id_canonico`. Entregar `doc_0201` onde se espera `2566535283` derruba a citação para erro, mesmo com a classe certa.

*Cobertura congelada em 1.014 registros. A métrica completa e o alinhamento por IoU estão implementados no **`kaggle_metric.py`**, incluído nesta aba (ver Overview → Evaluation).*

## Caminhos dos arquivos disponíveis

Procure pela pasta que tem esses arquivos quando necessario:

```text
desafio-jusbrasil-bracis-2026/
├── desafio1_bracis.db
├── goldenset_offsets.csv
├── json_to_submission.py
├── kaggle_metric.py
├── sample_submission.csv
└── txt/
    ├── gen_n1_001.txt … gen_n1_013.txt
    └── gen_n2_001.txt … gen_n2_013.txt
```
