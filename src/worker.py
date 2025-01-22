from BaseObject import BaseObject
class worker(BaseObject):
    def __init__(self, id):
        super().__init__(id)  # Inicializa atributos da classe base
        self.helmet_status_history = []  # Histórico de uso de capacete (True/False)

    def add_detection(self, bbox, helmet_status, frame_id):
        super().add_detection(bbox, frame_id)  # Chama o método da classe base para salvar bbox e frame_id
        self.helmet_status_history.append(helmet_status)

    def frame_area(self):
        return super().frame_area()
