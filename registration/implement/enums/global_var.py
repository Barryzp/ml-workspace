class Mode:
    matched = "matched"

class MatchUnit:
    #（1）原始的：divided分成若干个组，若干个块儿，每个算法在每个块儿内运行
    #（2）整体的：comp_one用一个算法跑，一个整体的块儿
    #（3）整体然后分层：comp_hierarchy（2）结果不行时就采用（3）
    divided = "divided"
    comp_one = "comp_one"
    comp_hierarchy = "comp_hierarchy"