import torch

def calculate_accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
    Returns:
    accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    _, predictions_indices = predictions.max(2)

    accuracy = (predictions_indices == targets).float().mean()

    return accuracy

# sample single character either using greedy approach
# or from categorical distribution with specified temperature


def sample_single(x, sampling='greedy', temperature=1.0):
    if sampling == 'greedy':
        output = torch.max(x, 0)[1].unsqueeze(0)
    elif sampling == 'random':
        if temperature > 0.0:
            x = x/temperature
        distr = torch.distributions.Categorical(logits=x)
        output = distr.sample(sample_shape=(1, 1))
    return output
