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
    