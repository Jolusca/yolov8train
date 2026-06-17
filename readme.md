
# Sistema de Detecção de Rostos com YOLOv8 e MTCNN

Este projeto visa a implementação de um sistema robusto de detecção facial, utilizando a arquitetura YOLOv8n para inferência em tempo real. A solução foi desenvolvida para ser leve, eficiente e de fácil reprodução, utilizando como base de dados o *FEI face database*. O grande diferencial desta implementação é a substituição de métodos de detecção tradicionais pelo MTCNN (Multi-task Cascaded Convolutional Networks), garantindo alta precisão na geração de *labels* e, consequentemente, um treinamento mais eficiente do modelo.

## Estrutura do Repositório
```text
.
├── dataset_original/      # Imagens puras para processamento
├── train/                 # Estrutura de dados para treinamento
├── val/                   # Estrutura de dados para validação
├── runs/                  # Logs e pesos (best.pt) do modelo
├── scripts/               # Scripts de processamento e teste
├── train.py               # Script principal de treinamento
└── dataset.yaml           # Configuração de classes e caminhos

```

## Instalação

1. Clone este repositório:
```bash
git clone <URL_DO_SEU_REPO>
cd yolov8train

```


2. Crie e ative o ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate

```


3. Instale as dependências:
```bash
pip install ultralytics opencv-python mtcnn scikit-learn tqdm

```



## Fluxo de Trabalho

### 1. Pré-processamento

Para garantir a qualidade dos rótulos, utilizamos o MTCNN.

* Coloque as imagens em `dataset_original/`.
* Execute o processador:
```bash
python scripts/processar_dataset.py

```


Este script normaliza as coordenadas e organiza os diretórios automaticamente.

### 2. Treinamento

Para iniciar o treinamento, edite o `dataset.yaml` com os caminhos corretos e execute:

```bash
python train.py

```

*Dica para Hardware Limitado:* Se ocorrer erro de memória, reduza `batch` para 2 e `imgsz` para 320 no `train.py`.

### 3. Inferência em Tempo Real

Para testar o modelo (`best.pt`) na sua webcam:

```bash
python scripts/testeYolo.py

```

## Considerações Acadêmicas

Os resultados obtidos demonstram a eficácia da arquitetura. Embora o uso do MTCNN tenha elevado a precisão espacial das caixas, o modelo é capaz de alcançar convergência estável rapidamente, apresentando alta velocidade de inferência (aprox. 11ms), tornando-o ideal para aplicações de reconhecimento facial que exigem baixa latência.
