from BaseObject import BaseObject
class Worker(BaseObject):
    def __init__(self, id):
        super().__init__(id)  # Inicializa atributos da classe base
        self.helmet_status_history = []  # Histórico de uso de capacete (True/False)

    def add_detection(self, bbox, helmet_status, frame_id):
        """
        Adiciona uma nova detecção ao histórico do trabalhador.
        
        Args:
            bbox (tuple): Coordenadas da bounding box (xmin, ymin, xmax, ymax).
            helmet_status (bool): True se estiver usando capacete, False caso contrário.
            frame_id (int): ID do frame atual.
        """
        super().add_detection(bbox, frame_id)  # Chama o método da classe base para salvar bbox e frame_id
        self.helmet_status_history.append(helmet_status)

    def has_no_helmet_consecutively(self, threshold=3):
        """
        Verifica se o trabalhador foi detectado sem capacete em frames consecutivos.
        
        Args:
            threshold (int): Número de frames consecutivos sem capacete para emitir alerta.

        Returns:
            bool: True se o trabalhador não estava usando capacete consecutivamente, False caso contrário.
        """
        # Pega os últimos `threshold` estados de capacete
        recent_status = self.helmet_status_history[-threshold:]
        # Retorna True apenas se todos forem False
        return len(recent_status) == threshold and all(not status for status in recent_status)
