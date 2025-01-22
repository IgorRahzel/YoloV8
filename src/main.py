import cv2
import os
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

# Criar e limpar diretórios necessários
os.makedirs('imgs', exist_ok=True)
os.makedirs('imgs/PessoasSemCapacete', exist_ok=True)
os.makedirs('imgs/Veiculos', exist_ok=True)
os.makedirs('alertas', exist_ok=True)
os.makedirs('alertas/pessoasSemCapacete', exist_ok=True)
os.makedirs('alertas/veiculos', exist_ok=True)

clear_directory('imgs/PessoasSemCapacete')
clear_directory('imgs/Veiculos')

# Limpar arquivos de log
with open('alertas/veiculos/alertas.log', 'w') as log_file:
    pass  # Limpa o conteúdo do arquivo
with open('alertas/pessoasSemCapacete/alertas.log', 'w') as log_file:
    pass  # Limpa o conteúdo do arquivo

# Carregar modelo YOLO e configurar vídeo
model = YOLO(r"model/best.pt")
video_path = 'ch5-cut.mp4'
cap = cv2.VideoCapture(video_path)

# Definir o instante inicial do vídeo (em segundos)
start_time_seconds = 150  # Por exemplo, 30 segundos
cap.set(cv2.CAP_PROP_POS_MSEC, start_time_seconds * 1000)  # Define a posição inicial em milissegundos


# Instanciar analisadores de vídeo
video_analyzer_vehicles = videoAnalyzer(object_type='veiculo')
video_analyzer_people = videoAnalyzer(object_type='pessoa')

current_frame = 0


while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Obter o timestamp do frame atual em milissegundos
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        # Converter o timestamp para minutos e segundos
        minutes = int(timestamp_ms // 60000)  # 1 minuto = 60000 milissegundos
        seconds = int((timestamp_ms % 60000) // 1000)  # O resto dividido por 1000 para segundos
        timestamp = f"{minutes:02d}:{seconds:02d}"  # Formato "minutos:segundos"

        # Realizar inferência com o modelo
        results = model(frame)
        frame = results[0].plot()

        # Analisar veículos e pessoas no frame
        video_analyzer_vehicles.video_analysis(frame,results,current_frame,timestamp)
        video_analyzer_people.video_analysis(frame,results,current_frame,timestamp)
      

        # Exibir saída em tempo real
        frame_resized = cv2.resize(frame, (640, 360))
        cv2.imshow('output', frame_resized)
        keyboard = cv2.waitKey(1)
        if keyboard == ord('q') or keyboard == 27:
            break
    else:
        break

    current_frame += 1

cap.release()
cv2.destroyAllWindows()

