'''
@file: MPNCOV.py
@author: Jiangtao Xie
@author: Peihua Li
Please cite the paper below if you use the code:
Peihua Li, Jiangtao Xie, Qilong Wang and Zilin Gao. Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization. IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR), pp. 947-955, 2018.
Peihua Li, Jiangtao Xie, Qilong Wang and Wangmeng Zuo. Is Second-order Information Helpful for Large-scale Visual Recognition? IEEE Int. Conf. on Computer Vision (ICCV),  pp. 2070-2078, 2017.
Copyright (C) 2018 Peihua Li and Jiangtao Xie
All rights reserved.
'''

import torch.nn as nn

__all__ = ['Newmodel', 'get_model']

from delphi.mpncov.base import Basemodel
from delphi.mpncov.mpncov import MPNCOV


class Newmodel(Basemodel):
    def __init__(self, num_classes, freezed_layer, model_dir, extract_feature_vector):
        super(Newmodel, self).__init__(model_dir, extract_feature_vector)
        self.representation = MPNCOV(input_dim=256)
        fc_input_dim = self.representation.output_dim

        self.classifier = nn.Linear(fc_input_dim, num_classes)
        index_before_freezed_layer = 0
        if freezed_layer:
            for m in self.features.children():
                if index_before_freezed_layer < freezed_layer:
                    self._freeze(m)
                index_before_freezed_layer += 1

    def _freeze(self, modules):
        for param in modules.parameters():
            param.requires_grad = False
        return modules


def get_model(num_classes, freezed_layer, model_dir, extract_feature_vector):
    _model = Newmodel(num_classes, freezed_layer, model_dir, extract_feature_vector)
    return _model
