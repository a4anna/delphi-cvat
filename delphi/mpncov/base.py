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

from delphi.mpncov.mpncovresnet import mpncovresnet50


class Basemodel(nn.Module):
    """Load backbone model and reconstruct it into three part:
       1) feature extractor
       2) global image representaion
       3) classifier
    """

    def __init__(self, model_dir, extract_feature_vector):
        super(Basemodel, self).__init__()
        basemodel = mpncovresnet50(model_dir)
        basemodel = self._reconstruct_mpncovresnet(basemodel)
        self.features = basemodel.features
        self.representation = basemodel.representation
        self.classifier = basemodel.classifier
        self.representation_dim = basemodel.representation_dim
        self.extract_feature_vector = extract_feature_vector

    def forward(self, x):
        x = self.features(x)
        x = self.representation(x)
        x = x.view(x.size(0), -1)

        if self.extract_feature_vector:
            return x
        else:
            return self.classifier(x)

    @staticmethod
    def _reconstruct_mpncovresnet(basemodel):
        model = nn.Module()
        model.features = nn.Sequential(*list(basemodel.children())[:-1])
        model.representation_dim = basemodel.layer_reduce.weight.size(0)
        model.representation = None
        model.classifier = basemodel.fc
        return model
