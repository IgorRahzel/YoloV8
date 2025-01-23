import cv2
import os
import numpy as np
from ultralytics import YOLO
from videoAnalyzer import videoAnalyzer

# Função para limpar o conteúdo de um diretório
def clear_directory(directory_path):
    if os.path.exists(directory_path):
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove arquivos ou links simbólicos
            elif os.path.isdir(file_path):
                os.rmdir(file_path)  # Remove subdiretórios vazios

def crop_frame(frame,ROI):
    # Extrair a região de interesse
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [ROI], (255, 255, 255))
    cropped = cv2.bitwise_and(frame, mask)
    x, y, w, h = cv2.boundingRect(ROI)
    final_crop = cropped[y:y+h, x:x+w]
    return final_crop
    
ROI = np.array([[0,310],[960,310],[1860,410],[1860,1080],[0,1080]], np.int32)

# Criar e limpar diretórios necessários
os.makedirs('imgs', exist_ok=True)
os.makedirs('imgs/PessoasSemCapacete', exist_ok=True)
os.makedirs('imgs/Veiculos', exist_ok=True)
os.makedirs('alertas', exist_ok=True)
os.makedirs('alertas/pessoasSemCapacete', exist_ok=True)
os.makedirs('alertas/veiculos', exist_ok=True)
os.makedirs('video_results', exist_ok=True)

clear_directory('imgs/PessoasSemCapacete')
clear_directory('imgs/Veiculos')

# Limpar arquivos de log
with open('alertas/veiculos/alertas.log', 'w') as log_file:
    pass  # Limpa o conteúdo do arquivo
with open('alertas/pessoasSemCapacete/alertas.log', 'w') as log_file:
    pass  # Limpa o conteúdo do arquivo

# Carregar modelo YOLO e configurar vídeo
model = YOLO(r"model/best (2).pt")
video_path = 'ch5-cut.mp4'
cap = cv2.VideoCapture(video_path)

# Definir o instante inicial do vídeo (em segundos)
start_time_seconds = 150
cap.set(cv2.CAP_PROP_POS_MSEC, start_time_seconds * 1000)  # Define a posição inicial em milissegundos

# Instanciar analisadores de vídeo
video_analyzer_vehicles = videoAnalyzer(object_type='veiculo')
video_analyzer_people = videoAnalyzer(object_type='pessoa')

current_frame = 0

# Configuração para salvar o vídeo de saída
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = 640
frame_height = 360
output_path = 'video_results/output_video.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para salvar o vídeo
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Controle para capturar até 1 minuto (60 segundos)
max_duration_seconds = 60
max_frames = fps * max_duration_seconds

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Obter o timestamp do frame atual em milissegundos
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        # Converter o timestamp para minutos e segundos
        minutes = int(timestamp_ms // 60000)
        seconds = int((timestamp_ms % 60000) // 1000)
        timestamp = f"{minutes:02d}:{seconds:02d}"  # Formato "minutos:segundos"

        # Extrair ROI
        frame = crop_frame(frame,ROI)

        # Redimensionar o frame para 640x360
        frame = cv2.resize(frame, (640, 360))

        # Realizar inferência com o modelo
        results = model(frame)
        frame = results[0].plot()

        # Analisar veículos e pessoas no frame
        video_analyzer_vehicles.video_analysis(frame, results, current_frame, timestamp)
        video_analyzer_people.video_analysis(frame, results, current_frame, timestamp)

        # Escrever o frame no vídeo de saída
        #out.write(frame)

        # Exibir o frame
        cv2.imshow('output', frame)

        # Parar ao atingir a duração máxima
        if current_frame >= max_frames:
            print("Tempo máximo atingido. Encerrando a captura.")
            break

        keyboard = cv2.waitKey(1)
        if keyboard == ord('q') or keyboard == 27:
            break
    else:
        break

    current_frame += 1

# Finalizar captura e liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Vídeo salvo em: {output_path}")
