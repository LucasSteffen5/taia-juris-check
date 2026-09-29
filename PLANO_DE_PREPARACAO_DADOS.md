Vamos construi um pipeline de preparacao de dados. Este pipeline deve ser totalmente deterministico e deve simular exatamente a mesma estretura que temos dos dados canonicos de C:\Users\lucas\OneDrive\Área de Trabalho\juris-check-TAIA\desafio-jusbrasil-bracis-2026 portanto nossa saida sera um .db um .csv e os proprios txt como estao. Antes de manipular qualquer coisa verifique se a codificacao, formatacao e fim de linhas estao seguindo o padrao imposto pelo DESAFIO.MD

Os dados brutos estão em C:\Users\lucas\OneDrive\Área de Trabalho\juris-check-TAIA\dataset_compartilhado

O resultado final é uma base confiavel totalmente feita por scripts, voce deve encontar uma maneira de criar de ponta ponta sem validacao humana totalmente automatizado. Note que em manifesto_reais temos 3501 arquivos que tem de fato uma citacao real portanto no minimo se cada arquivo tiver apenas uma citacao teremos 3501 citacoes reais. Mas sabemos que vamos ter mais




RULES:
Nao considere nada do que esta em PLANO DE SOLUCAO.md
Codigo simples e comentado em portugues
Menos codigo é melhor!
Pipeline deve ser facil de rodar com poucos comandos e sem parametros opcionais
Sempre revise se esta seguindo as regras do DESAFIO.MD
Reprodutibilidade é obrigatorio portanto devemos ter um produto final reprodutivel ponta a ponta
Prepare e documente para publicarmos no hunging face 
Para base que usaremos deixe creditos a  Eduardo Osti e Thiago Meirelles Ventura por ter disponibilizado os dados brutos para base