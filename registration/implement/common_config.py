

# 先预定义范围，超出这个编号的范围CT图像有点问题
AVALIABLE_RANGE = {
    1:[125, 970],
    2:[126, 1121],
    3:[125, 1162],
    4:[125, 1034],
    5:[125, 1170],
    6:[125, 1130],
    7:[125, 1200],
    8:[125, 1050],
    9:[125, 1150],
    10:[125, 1000],
    11:[125, 1120],
    12:[125, 1194],
    13:[125, 1114],

    # JPG格式的
    14:[185, 1030],
    15:[185, 1180],
    16:[185, 1345],
    17:[185, 1121],

    #BMP
    18:[185, 982],
    19:[185, 918],
    20:[185, 1030],
    21:[185, 1330],
    22:[185, 1345],
    23:[185, 1320],
    24:[185, 1110],
    25:[185, 1300],
    26:[185, 1054],
    27:[185, 1326],
    28:[185, 1326],
    29:[185, 1200],
    30:[185, 1200],
    31:[185, 1345],
    32:[185, 1200],
    33:[185, 1345],

    #TIF格式
    34:[185, 1158],
    35:[185, 1280],
    36:[185, 1200],
    37:[185, 1200],
    38:[185, 1345],
    39:[185, 1300],
    40:[185, 1250],
    41:[185, 1180],
    42:[185, 1200],
    43:[185, 1260],
    44:[185, 1260],
    45:[185, 839],
    46:[185, 1300],
    47:[185, 1250],
    48:[185, 1345],
    49:[185, 1345],
    50:[185, 1200],
    51:[185, 1345],
    52:[185, 1345],
    53:[185, 1250],
}

class CommonConfig:
    def __init__(self) -> None:
        self.config = AVALIABLE_RANGE

    def get_range(self, sample_id):
        return self.config[sample_id]