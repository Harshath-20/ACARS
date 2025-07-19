import torch
import torch.nn.functional as F

def contrastive_loss(z1, z2, temperature=0.5):
    """
    Compute contrastive loss between two batches of representations (z1, z2).
    Arguments:
        z1, z2: Tensor of shape (batch_size, embedding_dim)
        temperature: scaling factor for similarity
    Returns:
        Scalar loss value
    """
    # Normalize vectors
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate both batches
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)

    # Mask self-similarity
    batch_size = z1.size(0)
    labels = torch.arange(batch_size).to(z1.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(batch_size * 2, dtype=torch.bool).to(z1.device)
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

    # Similarity scores
    positives = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    negatives = torch.sum(torch.exp(similarity_matrix / temperature), dim=1)

    loss = -torch.log(positives / negatives[:batch_size])
    return loss.mean()
