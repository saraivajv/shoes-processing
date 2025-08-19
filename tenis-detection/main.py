import math

import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURAÇÕES ---
MODEL_PATH = "runs/detect/yolov8_tenis_detector/weights/best.pt"
VIDEO_PATH = "video_simulacao.mp4"
CONFIDENCE_THRESHOLD = 0.2
IOU_THRESHOLD = 0.5
MAX_DISTANCE = 50
FRAMES_TO_DISAPPEAR = 15
INITIALIZATION_FRAMES = 50
# --- FIM DAS CONFIGURAÇÕES ---

# Dicionário para armazenar o histórico de posições dos tênis rastreados
tracked_sneakers = {}
next_sneaker_id = 0

# Conjuntos para manter o controle dos IDs
sneaker_ids_seen = set()
collected_sneaker_ids = set()


# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(p1, p2):
    """Calcula a distância euclidiana entre dois pontos (x, y)."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def main():
    """
    Função principal que executa a detecção, rastreamento e contagem de tênis.
    """
    global next_sneaker_id

    # Carrega o modelo YOLOv8 treinado
    try:
        model = YOLO(MODEL_PATH)
        print("Modelo YOLO carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    # Abre o vídeo
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {VIDEO_PATH}")
        return

    # Pega o primeiro frame para seleção da ROI
    ret, first_frame = cap.read()
    if not ret:
        print("Não foi possível ler o primeiro frame do vídeo.")
        return

    # Permite que o usuário selecione a ROI
    roi = cv2.selectROI(
        "Selecione a ROI e pressione Enter",
        first_frame,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyWindow("Selecione a ROI e pressione Enter")

    x_roi, y_roi, w_roi, h_roi = roi

    # <<< NOVO: Contador de frames
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Corta a região de interesse diretamente do frame (mais eficiente que a máscara)
        frame_roi = frame[y_roi : y_roi + h_roi, x_roi : x_roi + w_roi]

        # Verifica se o corte da ROI é válido
        if frame_roi.size == 0:
            continue

        results = model(
            frame_roi,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        current_detections = []
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Converte as coordenadas relativas da ROI para coordenadas absolutas do frame
                abs_x1, abs_y1, abs_x2, abs_y2 = (
                    x1 + x_roi,
                    y1 + y_roi,
                    x2 + x_roi,
                    y2 + y_roi,
                )
                centroid_x = (abs_x1 + abs_x2) // 2
                centroid_y = (abs_y1 + abs_y2) // 2
                current_detections.append(
                    {
                        "centroid": (centroid_x, centroid_y),
                        "bbox": (
                            abs_x1,
                            abs_y1,
                            abs_x2,
                            abs_y2,
                        ),  # Salva as coordenadas absolutas
                    }
                )

        # --- LÓGICA DE RASTREAMENTO E ASSOCIAÇÃO ---

        used_cols = set()
        used_rows = set()

        if len(tracked_sneakers) > 0 and len(current_detections) > 0:
            track_ids = list(tracked_sneakers.keys())

            dist_matrix = np.zeros((len(track_ids), len(current_detections)))
            for i, track_id in enumerate(track_ids):
                last_centroid = tracked_sneakers[track_id]["centroids"][-1]
                for j, det in enumerate(current_detections):
                    dist_matrix[i, j] = euclidean_distance(
                        last_centroid, det["centroid"]
                    )

            rows, cols = np.unravel_index(
                np.argsort(dist_matrix, axis=None), dist_matrix.shape
            )

            used_rows = set()

            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue

                dist = dist_matrix[r, c]
                if dist < MAX_DISTANCE:
                    track_id = track_ids[r]
                    tracked_sneakers[track_id]["centroids"].append(
                        current_detections[c]["centroid"]
                    )
                    tracked_sneakers[track_id]["bbox"] = current_detections[c][
                        "bbox"
                    ]
                    tracked_sneakers[track_id]["disappeared"] = 0
                    used_rows.add(r)
                    used_cols.add(c)

        # --- LÓGICA DE COLETA E NOVOS TÊNIS ---

        if frame_count > INITIALIZATION_FRAMES:
            unmatched_track_ids = set(tracked_sneakers.keys()) - used_rows
            for track_id in unmatched_track_ids:
                tracked_sneakers[track_id]["disappeared"] += 1
                if (
                    tracked_sneakers[track_id]["disappeared"]
                    > FRAMES_TO_DISAPPEAR
                ):
                    collected_sneaker_ids.add(track_id)

        # Remove os tênis coletados da lista de rastreamento ativo
        for track_id in list(tracked_sneakers.keys()):
            if track_id in collected_sneaker_ids:
                del tracked_sneakers[track_id]

        # Registra novos tênis que não foram associados a nenhum existente (isso roda sempre)
        unmatched_det_indices = set(range(len(current_detections))) - used_cols
        for idx in unmatched_det_indices:
            det = current_detections[idx]
            tracked_sneakers[next_sneaker_id] = {
                "centroids": [det["centroid"]],
                "disappeared": 0,
                "bbox": det["bbox"],
            }
            sneaker_ids_seen.add(next_sneaker_id)
            next_sneaker_id += 1

        # --- VISUALIZAÇÃO ---

        cv2.rectangle(
            frame,
            (x_roi, y_roi),
            (x_roi + w_roi, y_roi + h_roi),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            "ROI",
            (x_roi, y_roi - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        for track_id, data in tracked_sneakers.items():
            x1, y1, x2, y2 = data["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Tenis ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        total_detectado = len(sneaker_ids_seen)
        total_coletados = len(collected_sneaker_ids)
        restantes = len(tracked_sneakers)

        if frame_count <= INITIALIZATION_FRAMES:
            cv2.putText(
                frame,
                "Inicializando... Contando objetos na cena",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        info_text_1 = f"Total Detectado (unico): {total_detectado}"
        info_text_2 = f"Coletados: {total_coletados}"
        info_text_3 = f"Restantes na Cena: {restantes}"

        cv2.putText(
            frame,
            info_text_1,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            info_text_2,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            info_text_3,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Detector e Contador de Tenis", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --- RELATÓRIO FINAL ---
    print("\n" + "=" * 30)
    print("RELATORIO FINAL DE COLETA")
    print("=" * 30)
    print(f"Total de tenis unicos detectados na ROI: {len(sneaker_ids_seen)}")
    print(f"Total de tenis coletados: {len(collected_sneaker_ids)}")
    # A contagem final de restantes é o total visto menos os coletados
    restantes_final = len(sneaker_ids_seen) - len(collected_sneaker_ids)
    print(f"Total de tenis restantes na cena: {restantes_final}")
    print("=" * 30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
