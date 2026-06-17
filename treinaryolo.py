from ultralytics import YOLO
import time

# Carrega a YOLOv8n com pesos pré-treinados para convergência rápida
model = YOLO("yolov8n.pt")

# Início da contagem do tempo
start_time = time.time()

# Treinamento ajustado para o dataset FEI
results = model.train(
    data='/home/lucas/git/yolov8train/dataset.yaml',
    epochs=60,
    patience=10,
    batch=8,  # Ajustado para o tamanho do seu dataset e capacidade de GPU
    imgsz=224,  # Resolução nativa das imagens FEI
    workers=2,  # Ajustado conforme seu hardware local
    pretrained=True, # Recomendado para não treinar do zero
    resume=False,
    single_cls=True, # Como você tem apenas uma classe (face)
    val=True,

    # Data Augmentation desabilitado conforme sua preferência
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.0,
    crop_fraction=1.0
)

# Tempo total de treinamento
end_time = time.time()
training_time = end_time - start_time

print(f"Tempo total de treinamento: {training_time:.2f} segundos")