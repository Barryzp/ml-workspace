import torch

class GlobalMatchDatas():

    def __init__(self) -> None:
        self.aim_slice_index = -1
        self.global_best_value = -1000
        self.global_best_position = torch.tensor([.0, .0, .0])
        self.global_best_img = None
        self.share_records_out = []
        self.threshold = 0.086
        
        self.stop_loop = False

    # 这个best_val都是正值
    def set_best(self, best_val, best_position, best_img, ct_slice_index):
        if best_val > self.global_best_value:
            self.global_best_value = best_val
            self.global_best_position = best_position
            self.global_best_img = best_img
            self.aim_slice_index = ct_slice_index
        
        if self.global_best_value > self.threshold:
            self.stop_loop = True

    def get_loop_state(self):
        return self.stop_loop
    
    def put_in_share_objects(self, ls):
        self.share_records_out.append(ls)