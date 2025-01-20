class worker:
    def __init__(self, person_id):
        self.person_id = person_id
        self.centroid_history = []  # Histórico das posições (centroides)
        self.helmet_status_history = []  # Histórico de uso de capacete (True/False)
        self.last_frame_seen = 0  # Último frame em que a pessoa foi detectada

    def add_detection(self, centroid, helmet_status, frame_id):
        """
        Adiciona uma nova detecção ao histórico da pessoa.
        
        Args:
            centroid (tuple): Coordenadas do centroide (cx, cy).
            helmet_status (bool): True se estiver usando capacete, False caso contrário.
            frame_id (int): ID do frame atual.
        """
        self.centroid_history.append(centroid)
        self.helmet_status_history.append(helmet_status)
        self.last_frame_seen = frame_id

    def has_no_helmet_consecutively(self, threshold=3):
        """
        Verifica se a pessoa foi detectada sem capacete em frames consecutivos.
        
        Args:
            threshold (int): Número de frames consecutivos sem capacete para emitir alerta.

        Returns:
            bool: True se a pessoa não estava usando capacete consecutivamente, False caso contrário.
        """
        # Pega os últimos `threshold` estados de capacete
        recent_status = self.helmet_status_history[-threshold:]
        # Retorna True apenas se todos forem False
        return len(recent_status) == threshold and all(not status for status in recent_status)
