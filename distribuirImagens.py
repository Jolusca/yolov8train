import os
import cv2
import shutil
from sklearn.model_selection import train_test_split

# 1. Configurar caminhos
base_path = '/home/lucas/git/yolov8train'
pasta_origem = os.path.join(base_path, 'dataset_original')

# Criar pastas de destino
for destino in ['train', 'val']:
    os.makedirs(os.path.join(base_path, destino, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_path, destino, 'labels'), exist_ok=True)

# 2. Listar todas as imagens da pasta original
imagens = [f for f in os.listdir(pasta_origem) if f.endswith('.jpg')]
train_imgs, val_imgs = train_test_split(imagens, test_size=0.2)

# 3. Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def processar_e_mover(lista_imgs, destino):
    for nome in lista_imgs:
        caminho_src = os.path.join(pasta_origem, nome)
        img = cv2.imread(caminho_src)
        
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Copiar imagem para a pasta de destino (train ou val)
        caminho_dst_img = os.path.join(base_path, destino, 'images', nome)
        shutil.copy(caminho_src, caminho_dst_img)
        
        # Criar label .txt correspondente
        caminho_dst_lbl = os.path.join(base_path, destino, 'labels', nome.replace('.jpg', '.txt'))
        with open(caminho_dst_lbl, 'w') as f:
            for (x, y, w, h) in faces:
                x_center = (x + w/2) / 640
                y_center = (y + h/2) / 480
                w_norm = w / 640
                h_norm = h / 480
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

# Executar a distribuição
processar_e_mover(train_imgs, 'train')
processar_e_mover(val_imgs, 'val')

print("Dataset organizado com sucesso!")