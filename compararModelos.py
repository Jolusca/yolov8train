from ultralytics import YOLO

# Caminhos dos modelos (ajuste conforme a pasta do seu train-X)
caminho_best = 'runs/detect/train-5/weights/best.pt'
caminho_last = 'runs/detect/train-5/weights/last.pt'

model_best = YOLO(caminho_best)
model_last = YOLO(caminho_last)

print("--- Avaliando o BEST.PT ---")
metrics_best = model_best.val(data='dataset.yaml')

print("\n--- Avaliando o LAST.PT ---")
metrics_last = model_last.val(data='dataset.yaml')

print(f"\nResultado Final:")
print(f"mAP50 Best: {metrics_best.box.map50:.4f}")
print(f"mAP50 Last: {metrics_last.box.map50:.4f}")