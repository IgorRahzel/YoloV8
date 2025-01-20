import cv2
import numpy as np
import os
from worker import worker

class videoAnalyzer:
    def __init__(self,object_type):
        if object_type not in ['veiculo','pessoa']:
            raise ValueError(f"Tipo de objeto inválido: {object_type}. Apenas 'pessoa' ou 'veiculo' são permitidos.")
        self.object_type = object_type
        self.people_recent_detections = []
        self.vehicles_recent_detections = []
        self.vehicle_id = 1
        self.person_id = 1


    def analyse(self,frame,results):
        if self.object_type == 'veiculo':
            self.veihcle_analysis(results,frame)

        elif self.object_type == 'pessoa':
            self.people_analysis(results,frame)
        
    
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
    def vehicle_analysis(self,frame,results,current_frame,timestamp):
        # Filter out old detections
        self.vehicles_recent_detections[:] = [(cx, cy, frame_id, vehicle_id)
                                               for cx, cy, frame_id, vehicle_id in self.vehicles_recent_detections
                                               if current_frame - frame_id <= 30
                                             ]

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
                for recent_vehicle in reversed(self.vehicles_recent_detections): # Iterate trough reversed list so last detectd vehicles are processed first
                    prev_cx, prev_cy,_, prev_id = recent_vehicle
                    distance = ((cx-prev_cx)**2 + (cy-prev_cy)**2)**0.5
                    print(f'distance:{distance}')
                    if (distance) < 80:
                        matched_id = prev_id
                        break
            
                # if id wasn't matched create new id
                if matched_id == None:
                    matched_id = self.vehicle_id
                    self.vehicle_id = self.vehicle_id + 1
            
                # Append vehicle information to self.vehicles_recent_detections = []
                self.vehicles_recent_detections.append((cx, cy, current_frame, matched_id))

                # Save image and create alert
                print(f'MatchedID: {matched_id}')
                print(f'VeiculoID: {self.vehicle_id}')
                if matched_id == self.vehicle_id - 1:
                    print('Entrou no Loop')
                    coordenadas = (xmin,ymin,xmax,ymax)
                    roi = self._get_roi(frame,coordenadas)
                    self._save_imgs('imgs/Veiculos',f'veiculo_{matched_id}.png',roi)
                    self._log_alerts(alert_type='veiculo',obj_id=matched_id,timestamp=timestamp,alert_path='alertas/veiculos/alertas.log')

            
    
    def people_analysis(self,frame,results,current_frame,timestamp):
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
        
        # Filter out old detections
        self.people_recent_detections[:] = [(cx, cy, frame_id, person_id)
                                               for cx, cy, frame_id, person_id in self.people_recent_detections
                                               if current_frame - frame_id <= 30
                                             ]
        
        # Anlyze every person in the frame
        for person in people_boxes:
            # Get person centroids
            xmin, ymin, xmax, ymax = person 
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            print(f'centroide:{cx.item(),cy.item()}')
            # Iterate over the self.people_recent_detections = [] and check if centroid is close to a previous one
            matched_id = None # Gives previous id to centroid that is closer than the thresholds
            min_dist = 200
            for recent_person in reversed(self.people_recent_detections): # Iterate trough reversed list so last detectd vehicles are processed first
                prev_cx, prev_cy,_, prev_id = recent_person
                distance = ((cx-prev_cx)**2 + (cy-prev_cy)**2)**0.5
                print(f'distance:{distance}')
                if (distance) < min_dist:
                    matched_id = prev_id
                    min_dist = distance
                    

            # if id wasn't matched create new id
            if matched_id == None:
                matched_id = self.person_id
                self.person_id = self.person_id + 1
            print(f'ID:{self.person_id}')
            # Append vehicle information to self.people_recent_detections
            self.people_recent_detections.append((cx, cy, current_frame, matched_id))

            # Calcular a região onde o capacete deve estar
            helmet_region_x_min = xmin
            helmet_region_x_max = xmax
            helmet_region_y_min = ymin - (ymax - ymin) // 2
            helmet_region_y_max = ymin

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
            print(f'Found Helmet:{found_helmet}')
            print(f'Matched_id:{matched_id} -- Person_id-1:{self.person_id-1}')
            if found_helmet == False and matched_id == self.person_id -1:
                print('Entrou no if')
                # salva imagem
                if len(helmet_boxes) != 0:
                    print(f'helmet_boxes:{helmet_boxes}')
                coordenadas = (xmin,ymin,xmax,ymax)
                roi = self._get_roi(frame,coordenadas)
                self._save_imgs('imgs/PessoasSemCapacete',f'pessoa_{matched_id}.png',roi)
                self._log_alerts(alert_type='pessoa',obj_id=matched_id,timestamp=timestamp,alert_path='alertas/pessoasSemCapacete/alertas.log')