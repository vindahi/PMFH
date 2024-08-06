import torch
import torch.nn as nn

class ConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, global_protos=None):
        device = torch.device('cuda' if features.is_cuda else 'cpu')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is None or global_protos is None:
            raise ValueError('Both `labels` and `global_protos` must be provided')

        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_count = contrast_count
            anchor_feature = torch.zeros_like(contrast_feature)
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Generate anchor_feature using global_protos and one-hot labels
        for i in range(batch_size * anchor_count):
            class_index = labels[i % batch_size].argmax().item()
            anchor_feature[i, :] = global_protos[class_index]

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create mask based on one-hot labels
        mask = torch.matmul(labels, labels.T).float().to(device)
        mask = mask.repeat(anchor_count, contrast_count)

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

