from ultralytics import YOLO
import time

# Carrega o modelo YOLOv8n
model = YOLO("yolov8n.pt")

# Início da contagem do tempo
start_time = time.time()

# Treinamento adaptado para uso eficiente da GPU
results = model.train(
    data='C:/Git/yolov8train/dataset.yaml',
    epochs=150,
    patience=30,
    batch=4,           
    imgsz=416,
    device='cpu',      # Altere para 'cpu' obrigatoriamente
    workers=4,         
    pretrained=True,
    single_cls=True,
    val=True,
    cache=True,
    
    # Data Augmentation desativado
    mosaic=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.0
)

# Tempo total de treinamento
end_time = time.time()
print(f"Tempo total de treinamento: {end_time - start_time:.2f} segundos")