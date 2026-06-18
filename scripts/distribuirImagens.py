import shutil
import random
from pathlib import Path

# Definindo caminhos
dataset_root = Path.cwd().parent
img_dir = dataset_root / 'dataset_original'
label_dir = dataset_root / 'labels'

# Criando estrutura de pastas para YOLO
(dataset_root / 'train/images').mkdir(parents=True, exist_ok=True)
(dataset_root / 'train/labels').mkdir(parents=True, exist_ok=True)
(dataset_root / 'val/images').mkdir(parents=True, exist_ok=True)
(dataset_root / 'val/labels').mkdir(parents=True, exist_ok=True)

imagens = list(img_dir.glob('*.jpg'))
random.shuffle(imagens)

# Definindo 80% para treino e 20% para validação
split = int(0.8 * len(imagens))
treino = imagens[:split]
validacao = imagens[split:]

def mover_arquivos(lista_imagens, destino_prefix):
    for img_path in lista_imagens:
        label_path = label_dir / f"{img_path.stem}.txt"
        
        if label_path.exists():
            shutil.copy(img_path, dataset_root / destino_prefix / 'images' / img_path.name)
            shutil.copy(label_path, dataset_root / destino_prefix / 'labels' / label_path.name)

mover_arquivos(treino, 'train')
mover_arquivos(validacao, 'val')

print("Divisão concluída: 80% treino e 20% validação.")