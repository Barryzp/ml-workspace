import torch
from utils.tools import Tools

class GlobalMatchDatas():

    def __init__(self, config) -> None:
        self.aim_slice_index = -1
        self.global_best_value = -1000
        self.iteration_count = 0
        self.global_best_position = torch.tensor([.0, .0, .0])
        self.global_best_img = None
        self.share_records_out = []
        self.threshold = config.matched_threshold
        
        self.stop_loop = False
        self.config = config

    # 这个best_val都是正值
    def set_best(self, best_val, best_position, best_img, ct_slice_index):
        if best_val > self.global_best_value:
            self.global_best_value = best_val
            self.global_best_position = best_position
            self.global_best_img = best_img
            self.aim_slice_index = ct_slice_index
            self.iteration_count += 1
            print(f"id: {self.iteration_count}; best val: {best_val}; ct_slice_index: {ct_slice_index}")
            # 把这张图片保存一下
            self.save_best_match()
        
        if self.global_best_value > self.threshold:
            self.stop_loop = True

    # 保存最佳图像
    def save_best_match(self):
        file_path = Tools.get_save_path(self.config)
        file_name = f"{self.iteration_count}_best_match_{self.aim_slice_index}.bmp"
        Tools.save_img(file_path, file_name, self.global_best_img)

    def get_loop_state(self):
        return self.stop_loop
    
    def put_in_share_objects(self, ls):
        self.share_records_out.append(ls)