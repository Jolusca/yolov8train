import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configurações de caminhos
base_dir = Path.cwd().parent
img_dir = base_dir / 'dataset_original'
label_dir = base_dir / 'labels'
label_dir.mkdir(parents=True, exist_ok=True)

# Caminhos do modelo Caffe (SSD)
prototxt_path = base_dir / "scripts" / "deploy.prototxt"
model_path = base_dir / "scripts" / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Carrega o detector
net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))

imagens = list(img_dir.glob('*.jpg'))

print("Iniciando rotulagem com SSD...")
for img_path in tqdm(imagens):
    img = cv2.imread(str(img_path))
    if img is None: continue
    
    h, w = img.shape[:2]
    
    # Prepara a imagem para a rede
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    # Considera a detecção com maior confiança
    melhor_conf = 0
    melhor_box = None
    
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.5: # Limiar de confiança
            if conf > melhor_conf:
                melhor_conf = conf
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                melhor_box = box.astype("int")
    
    if melhor_box is not None:
        startX, startY, endX, endY = melhor_box
        
        # Converte para formato YOLO
        x_center = ((startX + endX) / 2) / w
        y_center = ((startY + endY) / 2) / h
        box_w = (endX - startX) / w
        box_h = (endY - startY) / h
        
        # Salva o label
        label_path = label_dir / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

print("Processamento concluído. Dataset pronto para treino.")