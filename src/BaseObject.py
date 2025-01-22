import numpy as np
class BaseObject:
    def __init__(self,id):
        self.id = id
        self.bbox_history = []  # Histórico das posições (bounding boxes)
        self.last_frame_seen = 0  # Último frame em que a pessoa foi detectada
        self.frame = None
        self.frame_timestamp = None


    def add_detection(self, bbox, frame_id):
        self.bbox_history.append(bbox)
        self.last_frame_seen = frame_id
    

    def frame_area(self):
        if self.frame is None:
            return 0
        else:
            # Verifica se frame é um numpy array e obtém dimensões
            if isinstance(self.frame, np.ndarray):
                height, width = self.frame.shape[:2]
                return height * width
            else:
                raise ValueError("O atributo 'frame' não é um numpy array ou está em formato inválido.")
    