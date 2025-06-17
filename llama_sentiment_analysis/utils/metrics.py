import numpy as np
from llama_sentiment_analysis.config import CONFIG
import tensorflow as tf
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_metric

from llama_sentiment_analysis.llama import get_label_from_prediction


@tensorleap_custom_metric(name="metrics")
def calc_metrics(ground_truth: np.ndarray, attention_masks: np.ndarray, preds: np.ndarray):
    """`
    Calculate Accuracy, Precision, Recall, and F1 Score for NER.
    - Total metric score
    - Per each category class

    Parameters:
    true_labels (list of lists): True labels for each token.
    predicted_labels (list of lists): Predicted labels for each token.

    Returns:
    dictionary with:
        precision (float): Precision score
        recall (float): Recall score
        f1_score (float): F1 score

        [C]_precision, [C]_recall, [C]_f1_score for any category class C: [LOC, ORG, PER, MISC]

    """
    # attention_masks = attention_masks[None, ...]
    prediction, pred_token_text, last_token_index = get_label_from_prediction(attention_masks, preds)
    ground_truth = ground_truth[None, ...]
    ground_truth = tf.convert_to_tensor(ground_truth)
    prediction = tf.convert_to_tensor(prediction)

    prediction = tf.expand_dims(tf.argmax(prediction, axis=1), axis=1)

    metrics_names = ["accuracy", "precision"]
    metrics = {k: [] for k in metrics_names}

    def sample_precision_recall_f1(sample_ground_truth, sample_prediction):
        """ Calc the detailed metrics in sample level"""

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for true_label, pred_label in zip(sample_ground_truth, sample_prediction):
            true_label, pred_label = true_label.numpy(), pred_label.numpy()

            # Global All Classes TP, FN, ...
            if pred_label == true_label:  # Count only entity labels
                true_positives += 1
            elif pred_label != true_label:
                    false_positives += 1

        acc = tf.reduce_sum(tf.cast(sample_ground_truth == sample_prediction, tf.int32)) / max(1, len(sample_ground_truth))
        acc = acc.numpy().astype(np.float32)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        return {"accuracy": acc, "precision": precision}

    # Calc per each sample in batch
    for sample_ground_truth, sample_prediction in zip(ground_truth, prediction):
        dic_scores = sample_precision_recall_f1(tf.cast(sample_ground_truth, tf.int64), sample_prediction)
        for k in metrics:
            metrics[k].append(dic_scores[k])

    # Convert to tensorflow tensor
    for k, v in metrics.items():
        metrics[k] = np.array(v, dtype=np.float32)

    return metrics


def shannon_entropy(prob_dist):
    """
    Compute Shannon entropy of a probability distribution.

    :param prob_dist: List or array of probabilities.
    :return: Shannon entropy of the distribution.
    """
    prob_dist = np.array(prob_dist)
    prob_dist = prob_dist[prob_dist > 0]
    return -np.sum(prob_dist * np.log(prob_dist))