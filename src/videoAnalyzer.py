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
        self.vehicle_id = 1
        self.person_id = 1

        
    
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



    # Analyze the vehicles in the video
    def _vehicle_analysis(self,frame,results,current_frame,timestamp):
        
        # Iterate over the boxes
        for i,box in enumerate(results[0].boxes.xyxy):
            # Get bounding box coordinates
            xmin,ymin,xmax,ymax = box
            class_id = int(results[0].boxes.cls[i]) # Extract ID of the class
            class_name = results[0].names[class_id] # Extract class name
        

            # Check if class_name matche the object_type
            if class_name == self.object_type:
                # Get vehicle centroids
                cx = (xmin + xmax)/2
                cy = (ymin + ymax)/2
            
                # Iterate over the self.vehicles_recent_detections = [] and check if centroid is close to a previous one
                matched_id = None # Gives previous id to centroid that is closer than the thresholds
                min_dist = 80
                for _,prev_auto in self.automobile.items():
                    xmin_prev,ymin_prev,xmax_prev,ymax_prev = prev_auto.bbox_history[-1]
                    prev_cx = (xmin_prev + xmax_prev) / 2
                    prev_cy = (ymin_prev + ymax_prev) / 2
                    distance = ((cx-prev_cx)**2 + (cy-prev_cy)**2)**0.5
                    print(f'distance:{distance}')
                    if (distance) < min_dist:
                        matched_id = prev_auto.id
                        min_dist = distance
            
                # if id wasn't matched create new id
                if matched_id == None:
                    matched_id = self.vehicle_id
                    self.vehicle_id = self.vehicle_id + 1
                    self.automobile[matched_id] = vehicle(matched_id)
                
                #Store vehicle image:
                if len(self.people[matched_id].bbox_history) < 10:
                    coordenadas = (xmin,ymin,xmax,ymax)
                    person_roi = self._get_roi(frame,coordenadas)
                    self.automobile[matched_id].frame = person_roi
                    self.automobile[matched_id].timestamp = timestamp

            
                # Append vehicle information to self.vehicles_recent_detections = []
                self.automobile[matched_id].add_detection((xmin,ymin,xmax,ymax),current_frame)
            
    
    def _people_analysis(self,frame,results,current_frame,timestamp):
        people_boxes = [] # List to store identified boundig boxes for people
        helmet_boxes = [] # List to store identified boundig boxes for people
        # Iterate over bounding boxes founded in the frame
        for i,box in enumerate(results[0].boxes.xyxy):
            class_id = int(results[0].boxes.cls[i]) # Get class id of the BB
            class_name = results[0].names[class_id] #Get the class name of the BB
            # Check class that BB belongs to
            if class_name == 'pessoa':
                people_boxes.append(box)
            elif class_name == 'capacete':
                helmet_boxes.append(box)
        
        
        # Anlyze every person in the frame
        for person in people_boxes:
            # Get person centroids
            xmin, ymin, xmax, ymax = person 
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            print(f'centroide:{cx.item(),cy.item()}')
            # Iterate over the self.people.items() and check if centroid is close to a previous one
            matched_id = None # Gives previous id to centroid that is closer than the thresholds
            min_dist = 200
            for _,prev_person in self.people.items(): # Iterate trough reversed list so last detectd vehicles are processed first
                xmin_prev,ymin_prev,xmax_prev,ymax_prev = prev_person.bbox_history[-1]
                prev_cx = (xmin_prev + xmax_prev) / 2
                prev_cy = (ymin_prev + ymax_prev) / 2
                distance = ((cx-prev_cx)**2 + (cy-prev_cy)**2)**0.5
                print(f'distance:{distance}')
                if (distance) < min_dist:
                    matched_id = prev_person.id
                    min_dist = distance
                    

            # if id wasn't matched create new id
            if matched_id == None:
                matched_id = self.person_id
                self.person_id = self.person_id + 1
                self.people[matched_id] = worker(matched_id)
            print(f'ID:{self.person_id}')

            #Store person image:
            if len(self.people[matched_id].bbox_history) < 10:
                coordenadas = (xmin,ymin,xmax,ymax)
                person_roi = self._get_roi(frame,coordenadas)
                self.people[matched_id].frame = person_roi
                self.people[matched_id].timestamp = timestamp

            # Verificar se há capacete
            found_helmet = False
            for helmet in helmet_boxes:
                h_x_min, h_y_min, h_x_max, h_y_max = helmet
                #centroides do capacete
                h_cx = (h_x_max + h_x_min)/2
                h_cy = (h_y_max + h_y_min)/2
                #distancia do centroide do capacete para o centro da linha superior do bbox da pessoa
                helmet_centroid = np.array([h_cx,h_cy])
                person_centroid = np.array([cx,ymin])
                # Desenhar a linha entre os dois pontos (verde)
                cv2.line(frame, tuple(helmet_centroid.astype(int)), tuple(person_centroid.astype(int)), (0, 255, 0), 2)

                # Desenhar círculos nas extremidades da linha (azul)
                cv2.circle(frame, tuple(helmet_centroid.astype(int)), 5, (255, 0, 0), -1)  # Ponto azul na extremidade do capacete
                cv2.circle(frame, tuple(person_centroid.astype(int)), 5, (255, 0, 0), -1)  # Ponto azul na extremidade da pessoa
                dist_helmet2person = np.linalg.norm(helmet_centroid - person_centroid)
                print(f'Dist_helmet2person: {dist_helmet2person}')
                if dist_helmet2person < 100:
                    found_helmet = True
                    break
            
            # Update helmet history
            self.people[matched_id].add_detection((xmin,ymin,xmax,ymax), found_helmet, current_frame)
    
    

    def video_analysis(self,frame,results,current_frame,timestamp):
        if self.object_type == 'veiculo':
            self._vehicle_analysis(frame,results,current_frame,timestamp)
        elif self.object_type == 'pessoa':
            self._people_analysis(frame,current_frame,timestamp)
        
        self.create_alert(current_frame)
            

    def _create_alert_vehicles(self,current_frame):
        to_remove = []
        for vehicle_id,vehicle in self.automobile.items():
            if current_frame - vehicle.last_frame_seen > 30:
                roi = vehicle.frame
                self._save_imgs('imgs/Veiculos', f'veiculo_{vehicle_id}.png', roi)
                self._log_alerts(alert_type='vehicle', obj_id=vehicle_id, timestamp=vehicle.timestamp, alert_path='alertas/veiculos/alertas.log')

                # Adicionar o ID da pessoa à lista de remoção
                to_remove.append(vehicle_id)

        # Remover pessoas que saíram do vídeo
        for vehicle_id in to_remove:
            del self.people[vehicle_id]


    
    def _create_alert_people(self,current_frame):
        # Verificar se há pessoas que não foram vistas recentemente
        to_remove = []
        for person_id, person in self.people.items():
            if current_frame - person.last_frame_seen > 30:  # Exemplo: 30 frames sem ser detectado
                # Pessoa saiu do vídeo, calcular estatísticas
                total_detections = len(person.helmet_status_history)
                no_helmet_count = person.helmet_status_history.count(False)
                helmet_ratio = no_helmet_count / total_detections if total_detections > 0 else 0

                print(f"Pessoa {person_id} saiu do vídeo. Razão sem capacete: {helmet_ratio:.2f}")

                if helmet_ratio > 0.80:  # Exemplo: Threshold de 50% sem capacete
                    # Salvar imagem da última posição detectada
                    last_bbox = person.bbox_history[-1]
                    xmin, ymin, xmax, ymax = last_bbox
                    roi = person.frame
                    self._save_imgs('imgs/PessoasSemCapacete', f'pessoa_{person_id}.png', roi)
                    self._log_alerts(alert_type='pessoa', obj_id=person_id, timestamp=person.timestamp, alert_path='alertas/pessoasSemCapacete/alertas.log')

                # Adicionar o ID da pessoa à lista de remoção
                to_remove.append(person_id)

        # Remover pessoas que saíram do vídeo
        for person_id in to_remove:
            del self.people[person_id]


    def create_alert(self,current_frame):
        if self.object_type == 'veiculos':
            self._create_alert_vehicles(current_frame)
        elif self.object_type == 'pessoa':
            self._create_alert_people(current_frame)