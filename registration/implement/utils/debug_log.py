


# 日志打印管理器
class Logger():
    def __init__(self, config) -> None:
        self.log_toggle = config.show_log
    
    def log(self, sth):
        if self.log_toggle:
            print(sth)