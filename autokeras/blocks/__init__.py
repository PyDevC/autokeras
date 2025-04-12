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

import keras
import tensorflow as tf

from autokeras.blocks.basic import (
    BertBlock,
    ConvBlock,
    DenseBlock,
    EfficientNetBlock,
    ResNetBlock,
    RNNBlock,
    XceptionBlock,
)

from autokeras.blocks.heads import ClassificationHead, RegressionHead

from autokeras.blocks.preprocessing import ImageAugmentation, Normalization

from autokeras.blocks.reduction import (
    Flatten,
    Merge,
    SpatialReduction,
    TemporalReduction,
)

from autokeras.blocks.wrapper import (
    GeneralBlock,
    ImageBlock,
    TextBlock,
)

from autokeras.utils import utils


def serialize(obj):
    return utils.serialize_keras_object(obj)


def deserialize(config, custom_objects=None):
    return utils.deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
    )
