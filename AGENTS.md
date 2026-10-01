# Agent instructions

## Protected artifact

Treat `DESAFIO.md` as an immutable, user-approved artifact. Do not edit, rewrite, reformat, delete, or regenerate it. If a later task appears to require changing it, first ask the user to explicitly remove this protection.

## Dados oficiais

Não use os dados de `data/desafio-jusbrasil-bracis-2026` para treinamento, exceto a autorização específica desta campanha de fine-tuning para os dez documentos abaixo e suas anotações:

- N1: `gen_n1_001`, `gen_n1_005`, `gen_n1_006`, `gen_n1_008`, `gen_n1_013`.
- N2: `gen_n2_001`, `gen_n2_002`, `gen_n2_006`, `gen_n2_011`, `gen_n2_013`.

Os outros 16 documentos foram reservados para a avaliação final da campanha. Qualquer uso adicional de dados oficiais para treino exige nova autorização explícita do usuário.

## Execução

Tudo deste repositório deve estar dockerizado e ser fácil de rodar com poucos comandos.
