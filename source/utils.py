# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s
import time
from scipy import stats
import numpy as np


def c_index(y_true, y_pred):
    """
    Concordance Index Function

    Args:
    - y_trues: true values
    - y_pred: predicted values

    """

    matrix_pred = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    matrix_pred = tf.cast(matrix_pred == 0.0, tf.float32) * 0.5 + tf.cast(matrix_pred > 0.0, tf.float32)

    matrix_true = tf.subtract(tf.expand_dims(y_true, -1), y_true)
    matrix_true = tf.cast(matrix_true > 0.0, tf.float32)

    matrix_true_position = tf.where(tf.equal(matrix_true, 1))

    matrix_pred_values = tf.gather_nd(matrix_pred, matrix_true_position)

    # If equal to zero then it returns zero, else return the result of the division
    result = tf.where(tf.equal(tf.reduce_sum(matrix_pred_values), 0), 0.0,
                      tf.reduce_sum(matrix_pred_values) / tf.reduce_sum(matrix_true))

    return result


def min_max_scale(data):
    data_scaled = (data - np.min(data)) / ((np.max(data) - np.min(data)) + 1e-05)

    return data_scaled

import os
def inference_metrics(model, data, batch_size=16):
    """
    Prediction Efficiency Evaluation Metrics with Batching

    Args:
    - model: trained model
    - data: [protein data, smiles data, kd values]
    - batch_size: size of each batch for inference
    """

    # Set TF_GPU_ALLOCATOR to 'cuda_malloc_async'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    # Enable Memory Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Initialize prediction and timing
    num_samples = data[0].shape[0]
    pred_values = []
    start = time.time()

    # Process data in batches
    for i in range(0, num_samples, batch_size):
        batch_protein = data[0][i:i + batch_size]
        batch_smiles = data[1][i:i + batch_size]
        batch_pred = model([batch_protein, batch_smiles], training=False)
        pred_values.extend(batch_pred)

    # Convert predictions to numpy array and calculate inference time
    pred_values = np.array(pred_values)
    end = time.time()
    inf_time = end - start

    # Calculate metrics
    metrics = {
        'MSE': mse(data[2], pred_values),
        'RMSE': mse(data[2], pred_values, squared=False),
        'CI': c_index(data[2], pred_values).numpy(),
        'R2': r2s(data[2], pred_values),
        'Spearman': stats.spearmanr(data[2].numpy(), pred_values)[0],
        'Time': inf_time
    }

    return metrics
