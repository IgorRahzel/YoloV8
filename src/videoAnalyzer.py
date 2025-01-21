import cv2
import numpy as np
import os
from worker import worker
from vehicle import vehicle

class videoAnalyzer:
    def __init__(self,object_type):
        if object_type not in ['veiculo','pessoa']:
            raise ValueError(f"Tipo de objeto inválido: {object_type}. Apenas 'pessoa' ou 'veiculo' são permitidos.")
        self.object_type = object_type
        self.people = {}
        self.automobile= {}
        self.vehicle_id = [1]
        self.person_id = [1]

        
    
    def _get_roi(self,frame,coordenadas):
        x_min, y_min, x_max, y_max = map(int, coordenadas)
        h, w, _ = frame.shape
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        roi = frame[y_min:y_max, x_min:x_max]
        return roi
    
 

    def _log_alerts(self, alert_type, obj_id, timestamp, alert_path):

        # Garante que o diretório do arquivo de log exista
        os.makedirs(os.path.dirname(alert_path), exist_ok=True)

        # Formata a mensagem de log
        log_message = f"{alert_type} ID {obj_id} detectado no timestamp {timestamp}\n"

        # Verifica se a mensagem já existe no arquivo de log
        if os.path.exists(alert_path):
            with open(alert_path, "r") as log_file:
                existing_logs = log_file.readlines()
                # Confere se o ID já foi registrado
                if any(f"{alert_type} ID {obj_id}" in log for log in existing_logs):
                    return  # Não grava a mensagem se ela já existe

        # Grava a mensagem no arquivo de log
        with open(alert_path, "a") as log_file:
            log_file.write(log_message)



    def _save_imgs(self, dir_path, img_name, img):
        # Cria o diretório, se ainda não existir
        os.makedirs(dir_path, exist_ok=True)
        img_path = os.path.join(dir_path, img_name)
        # Verifica se o arquivo já existe antes de salvar
        if not os.path.exists(img_path):
            cv2.imwrite(img_path, img)

    
    def _object_analysis(self, frame, results, current_frame, timestamp, object_type, storage_dict, object_class, id_counter, threshold, check_helmet=False):
    
        object_boxes = []  # Bounding boxes do tipo de objeto
        helmet_boxes = []  # Bounding boxes de capacetes (somente para pessoas)

        # Identificar bounding boxes relevantes no frame
        for i, box in enumerate(results[0].boxes.xyxy):
            class_id = int(results[0].boxes.cls[i])
            class_name = results[0].names[class_id]
            if class_name == object_type:
                object_boxes.append(box)
            elif check_helmet and class_name == 'capacete':
                helmet_boxes.append(box)

        # Processar cada bounding box do objeto
        for obj_box in object_boxes:
            xmin, ymin, xmax, ymax = obj_box
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

            # Verificar correspondência de ID baseado na distância
            matched_id = None
            min_dist = threshold
            for _, prev_obj in storage_dict.items():
                xmin_prev, ymin_prev, xmax_prev, ymax_prev = prev_obj.bbox_history[-1]
                prev_cx, prev_cy = (xmin_prev + xmax_prev) / 2, (ymin_prev + ymax_prev) / 2
                distance = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                if distance < min_dist:
                    matched_id = prev_obj.id
                    min_dist = distance

            # Criar novo ID se não houver correspondência
            if matched_id is None:
                matched_id = id_counter[0]
                id_counter[0] += 1
                storage_dict[matched_id] = object_class(matched_id)
            
            print(f'matched_id: {matched_id}')

            # Armazenar ROI e timestamp
            if len(storage_dict[matched_id].bbox_history) < 10:
                roi = self._get_roi(frame, (xmin, ymin, xmax, ymax))
                storage_dict[matched_id].frame = roi
                storage_dict[matched_id].timestamp = timestamp

            # Atualizar informações do objeto
            if check_helmet:
                # Verificar capacete se necessário
                found_helmet = False
                for helmet in helmet_boxes:
                    h_xmin, h_ymin, h_xmax, h_ymax = helmet
                    h_cx, h_cy = (h_xmax + h_xmin) / 2, (h_ymax + h_ymin) / 2
                    dist_helmet2person = ((cx - h_cx) ** 2 + (ymin - h_cy) ** 2) ** 0.5
                    if dist_helmet2person < 100:
                        found_helmet = True
                        break
                storage_dict[matched_id].add_detection((xmin, ymin, xmax, ymax), found_helmet, current_frame)
            else:
                storage_dict[matched_id].add_detection((xmin, ymin, xmax, ymax), current_frame)

    
    

    def video_analysis(self,frame,results,current_frame,timestamp):
        if self.object_type == 'veiculo':
            self._object_analysis(  frame=frame,
                                    results=results,
                                    current_frame=current_frame,
                                    timestamp=timestamp,
                                    object_type=self.object_type,
                                    storage_dict=self.automobile,
                                    object_class=vehicle,
                                    id_counter=self.vehicle_id,
                                    threshold=80,
                                    check_helmet=False
                                )

        elif self.object_type == 'pessoa':
            self._object_analysis(
                                    frame=frame,
                                    results=results,
                                    current_frame=current_frame,
                                    timestamp=timestamp,
                                    object_type='pessoa',
                                    storage_dict=self.people,
                                    object_class=worker,
                                    id_counter=self.person_id,
                                    threshold=200,
                                    check_helmet=True
                                )

        
        self.create_alert(current_frame)
            
    
   

    def create_obj_alert(self,current_frame,storage_dict):
         # Verificar se há pessoas que não foram vistas recentemente
        to_remove = []
        for obj_id, obj in storage_dict.items():
            if current_frame - obj.last_frame_seen > 30:  # Exemplo: 30 frames sem ser detectado
                if self.object_type == 'pessoa':
                    # Pessoa saiu do vídeo, calcular estatísticas
                    total_detections = len(obj.helmet_status_history)
                    no_helmet_count = obj.helmet_status_history.count(False)
                    helmet_ratio = no_helmet_count / total_detections if total_detections > 0 else 0

                    print(f"Pessoa {obj_id} saiu do vídeo. Razão sem capacete: {helmet_ratio:.2f}")

                    if helmet_ratio > 0.80:  # Exemplo: Threshold de 50% sem capacete
                        # Salvar imagem da última posição detectada
                        roi = obj.frame
                        self._save_imgs('imgs/PessoasSemCapacete', f'pessoa_{obj_id}.png', roi)
                        self._log_alerts(alert_type='pessoa', obj_id=obj_id, timestamp=obj.timestamp, alert_path='alertas/pessoasSemCapacete/alertas.log')
                
                elif self.object_type == 'veiculo':
                    roi = obj.frame
                    self._save_imgs('imgs/Veiculos', f'veiculo_{obj_id}.png', roi)
                    self._log_alerts(alert_type='vehicle', obj_id= obj_id, timestamp=vehicle.timestamp, alert_path='alertas/veiculos/alertas.log')

                # Adicionar o ID da pessoa à lista de remoção
                to_remove.append(obj_id)
            
        # Remover pessoas que saíram do vídeo
        for obj_id in to_remove:
            del storage_dict[obj_id]


    def create_alert(self,current_frame):
        if self.object_type == 'veiculos':
            self.create_obj_alert(current_frame,self.automobile)
        elif self.object_type == 'pessoa':
            self.create_obj_alert(current_frame,self.people)