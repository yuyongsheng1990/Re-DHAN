# -*- coding: UTF-8 -*-
# @Project -> File: run_offline model.py -> KPGNN
# @Time: 4/8/23 18:47 
# @Author: Yu Yongsheng
# @Description:

"""
    KPGNN: 2-layer GAT. 就是没有RL的intra_agg in FinEvent.
    每一个meta-path 计算一次loss，将三次loss相加求和，进行bp
"""