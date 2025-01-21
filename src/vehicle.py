from BaseObject import BaseObject
class vehicle(BaseObject):
    def __init__(self,id):
        super().__init__(id)
    

    def add_detection(self, bbox, frame_id):
        return super().add_detection(bbox, frame_id)
