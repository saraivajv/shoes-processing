import torch
from ultralytics import YOLO


def main():
    """
    Função principal para treinar o modelo YOLOv8.
    """
    # --- Verificação de GPU ---
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Treinamento será executado na GPU: {device_name}")
        device = 0
    else:
        print(
            "GPU não encontrada. O treinamento será executado na CPU (muito mais lento)."
        )
        device = "cpu"

    # --- Carregamento do Modelo ---
    model = YOLO("yolov8n.pt")

    # --- Treinamento do Modelo ---
    print("Iniciando o treinamento do modelo...")
    results = model.train(
        # --- Configurações do Dataset ---
        data="data.yaml",
        # --- Configurações de Treinamento ---
        epochs=75,
        imgsz=640,
        batch=8,
        # --- Configurações de Hardware e Salvamento ---
        device=device,
        name="yolov8_tenis_detector",
    )

    print("-" * 30)
    print("Treinamento concluído com sucesso!")
    print(
        f"O melhor modelo foi salvo em: runs/detect/{results.save_dir.name}/weights/best.pt"
    )
    print("-" * 30)


if __name__ == "__main__":
    main()
