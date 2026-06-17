import os

# Caminho para as pastas que contêm os labels
for pasta in ['train/labels', 'val/labels']:
    caminho_labels = os.path.join('/home/lucas/git/yolov8train', pasta)
    
    for arquivo in os.listdir(caminho_labels):
        caminho_txt = os.path.join(caminho_labels, arquivo)
        
        # Verifica se o arquivo está vazio
        if os.path.getsize(caminho_txt) == 0:
            print(f"Removendo arquivo vazio: {arquivo}")
            os.remove(caminho_txt)
            
            # Remove a imagem correspondente também
            nome_img = arquivo.replace('.txt', '.jpg')
            caminho_img = caminho_txt.replace('labels', 'images').replace('.txt', '.jpg')
            if os.path.exists(caminho_img):
                os.remove(caminho_img)