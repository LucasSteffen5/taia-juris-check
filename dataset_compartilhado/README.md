# Dataset Jurídico Estruturado e Categorizado (5.000 Peças Auditadas)
## Desafio Caça-Alucinações · BRACIS 2026 × Jusbrasil

Este pacote contém **5.000 documentos jurídicos** rigorosamente auditados arquivo por arquivo para garantir total correspondência entre o texto e a pasta onde está armazenado.

### Arquitetura de Pastas e Contagem Exata:

```text
dataset_compartilhado/
│
├── reais/                          # 3.500 Peças Reais (validadas pelo cabeçalho do Tribunal)
│   ├── stj/                        #  1137 acórdãos do Superior Tribunal de Justiça
│   ├── tcu/                        #  2305 acórdãos e tomadas de contas do Tribunal de Contas da União
│   └── outros_tribunais/           #    58 acórdãos (TSE, STM, TST, TRFs, TJs e LeNER-Br)
│
├── inventados/                     # 1.500 Peças Sintéticas (validadas por Área do Direito Estrita)
│   ├── administrativo/             #   509 peças de Direito Administrativo (Licitações, Pareceres, OSC)
│   ├── civel/                      #   178 peças de Direito Cível (Contratos, Responsabilidade Civil, Posse)
│   ├── consumidor/                 #   195 peças de Direito do Consumidor (CDC, Vício do Produto)
│   ├── trabalhista/                #   190 peças de Direito Trabalhista (CLT, Verbas, Horas Extras)
│   ├── tributario/                 #   117 peças de Direito Tributário (CTN, Impostos, Execuções Fiscais)
│   ├── penal/                      #   111 peças de Direito Penal (CP, CPP, Crimes)
│   ├── ambiental/                  #   100 peças de Direito Ambiental (IBAMA, APP, Dano Ambiental)
│   └── empresarial/                #   100 peças de Direito Empresarial (Sociedades, Falências, Quotas)
│
├── manifesto_reais.csv             # Catálogo com caminhos relativos e tribunais
├── manifesto_inventados.csv        # Catálogo com caminhos relativos, áreas e tipos
└── README.md                       # Este guia de uso
```

### Como Carregar em Python
```python
import glob
import pandas as pd

# 1. Usando glob recursivo
arquivos_stj = glob.glob("reais/stj/*.txt")
arquivos_consumidor = glob.glob("inventados/consumidor/*.txt")
todos_reais = glob.glob("reais/**/*.txt", recursive=True)

# 2. Usando os manifestos CSV
df_reais = pd.read_csv("manifesto_reais.csv")
df_inventados = pd.read_csv("manifesto_inventados.csv")
print(df_reais['tribunal'].value_counts())
print(df_inventados['area_direito'].value_counts())
```
