from typing import Any, Callable, Optional, Tuple
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import time

import numpy as np
from PIL import Image
import sys

BATCHSIZE = 180

def transformTensor(arr):
    arr = tf.cast(arr, dtype=tf.float32)
    arr = tf.math.subtract(arr, tf.math.reduce_mean(arr))
    arr = tf.math.divide(arr, tf.math.reduce_std(arr)) 
    arr = tf.reshape(arr, [3, 300, 300, BATCHSIZE])
    out = tf.zeros([0, 300, 300, 3])
    for i in range(BATCHSIZE):
        temp = arr[:, :, :, i]
        temp = tf.stack([temp, temp, temp])
        temp = tf.reshape(temp, [3, 300, 300, 3])
        out = tf.concat([out, temp], axis=0)
    out = tf.reshape(out, [BATCHSIZE, 3, 300, 300, 3])
    #out = tf.transpose(out, [3, 1, 2, 0])
    out1 = tf.reshape(out[:, 0, :, :, :], [BATCHSIZE, 300, 300, 3])
    out2 = tf.reshape(out[:, 1, :, :, :], [BATCHSIZE, 300, 300, 3])
    out3 = tf.reshape(out[:, 2, :, :, :], [BATCHSIZE, 300, 300, 3])
    return out1, out2, out3

def transformLabel(l):
    l = tf.cast(tf.math.subtract(l, 1), dtype=tf.uint8)
    l = tf.one_hot(l, 6)
    return l

class CustomTFDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        X_generator,
        y_generator,
        *,
        transform: Optional[Callable],
        target_transform: Optional[Callable] = None,
    ) -> None:
        
        self.X_generator = X_generator
        self.y_generator = y_generator
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.X_generator)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        X_batch = tf.convert_to_tensor(self.X_generator[idx].data)
        y_batch = tf.convert_to_tensor(self.y_generator[idx].data)
        if self.transform:
            X_batch = self.transform(X_batch)
        if self.target_transform:
            y_batch = self.target_transform(y_batch)
        return X_batch, y_batch

