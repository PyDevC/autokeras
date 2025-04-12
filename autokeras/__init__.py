# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import keras_nlp

from autokeras.auto_model import AutoModel

from autokeras.blocks import (
    BertBlock,
    ClassificationHead,
    ConvBlock,
    DenseBlock,
    EfficientNetBlock,
    Flatten,
    ImageAugmentation,
    ImageBlock,
    Merge,
    Normalization,
    RegressionHead,
    ResNetBlock,
    RNNBlock,
    SpatialReduction,
    TemporalReduction,
    TextBlock,
    XceptionBlock,
)

from autokeras.engine.block import Block
from autokeras.engine.head import Head
from autokeras.engine.node import Node
from autokeras.keras_layers import CastToFloat32, ExpandLastDim

from autokeras.nodes import (
    ImageInput,
    Input,
    TextInput,
)

from autokeras.tasks import (
    ImageClassifier,
    ImageRegressor,
    TextClassifier,
    TextRegressor,
)

from autokeras.tuners import (
    BayesianOptimization,
    Greedy,
    Hyperband,
    RandomSearch,
)

from autokeras.utils.io_utils import (
    image_dataset_from_directory,
    text_dataset_from_directory,
)

__version__ = "2.1.0dev"

CUSTOM_OBJECTS = {
    "BertPreprocessor": keras_nlp.models.BertPreprocessor,
    "BertBackbone": keras_nlp.models.BertBackbone,
    "CastToFloat32": CastToFloat32,
    "ExpandLastDim": ExpandLastDim,
}
