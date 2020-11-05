import io
import os
from logzero import logger
from pathlib import Path
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms, models

from delphi.mpncov.model_init import get_model
from delphi.svm.feature_cache import FeatureCache
from delphi.utils import get_example_key, log_exceptions

BATCH_SIZE = 8


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FeatureProvider(object):

    def __init__(self, feature_extractor: str, cache: FeatureCache):
        self.feature_extractor = feature_extractor
        self.cache = cache

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._model = self._get_model().to(self.device, non_blocking=True)
        self._model.eval()

        self._preprocessor = self._get_preprocessor()

    def get_result_key_name(self, name: str):
        return '{}_{}'.format(name, self.feature_extractor)

    def get_result_key_content(self, content: bytes):
        return '{}_{}'.format(get_example_key(content), self.feature_extractor)

    def get_cached_vector(self, result_key: str) -> Optional[List[float]]:
        return self.cache.get(result_key)

    def preprocess(self, content: bytes) -> torch.Tensor:
        try:
            image = Image.open(io.BytesIO(content))

            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            image = Image.fromarray(np.random.randint(0,255,(256,256,3)),'RGB')

        return self._preprocessor(image)

    def cache_and_get(self, result_keys: List[str], tensor: torch.Tensor, expire: bool) -> Dict[str, List[float]]:
        with torch.no_grad():
            feature_vectors = F.normalize(self._model(tensor))

        results = dict()
        for i in range(len(result_keys)):
            results[result_keys[i]] = feature_vectors[i].data.tolist()

        self.cache.put(results, expire)
        return results

    def _get_model(self):
        if self.feature_extractor == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier = Identity()
            return model
        elif self.feature_extractor == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.fc = Identity()
            return model
        elif self.feature_extractor == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.fc = Identity()
            return model
        elif self.feature_extractor == 'mpncov_resnet50':
            # Download MPNCOV-RESNET50 model from
            # https://drive.google.com/file/d/19TWen7p9UDyM0Ueu9Gb22NtouR109C6j/view
            model_dir = Path(os.environ.get('DELPHI'), 'models')
            return get_model(2, None, model_dir, True)
        else:
            raise NotImplementedError('unknown feature extractor ' + self.feature_extractor)

    def _get_preprocessor(self):
        return transforms.Compose([
            transforms.Resize(512 if self.feature_extractor == 'mpncov_resnet50' else 256),
            transforms.CenterCrop(448 if self.feature_extractor == 'mpncov_resnet50' else 224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


_feature_provider: FeatureProvider


def get_worker_feature_provider() -> FeatureProvider:
    global _feature_provider
    return _feature_provider


@log_exceptions
def set_worker_feature_provider(feature_extractor: str, cache: FeatureCache) -> None:
    global _feature_provider
    _feature_provider = FeatureProvider(feature_extractor, cache)
