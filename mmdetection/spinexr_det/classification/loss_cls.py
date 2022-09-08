import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, device, class_weight, num_classes, gamma=2):
        """
        class weight should be a list. 
        """
        super().__init__()
        self.device = device
        self.class_weight = torch.tensor(class_weight, device=device)
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        yTrueOnehot = torch.zeros(y_true.size(
            0), self.num_classes, device=self.device)
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)
        y_pred = torch.clamp(y_pred, min=1e-6, max=1-1e-6)

        focal = - yTrueOnehot * (1-y_pred)**self.gamma * \
            torch.log(y_pred) * self.class_weight
        active = yTrueOnehot * (1 - y_pred) * self.class_weight
        bce = - yTrueOnehot * torch.log(y_pred) * self.class_weight
        loss = torch.sum(focal) + torch.sum(active) + torch.sum(bce)
        return loss / (torch.sum(self.class_weight) * y_true.size(0))


class ActiveContourLoss(nn.Module):
    def __init__(self, device, class_weight, num_classes):
        """
        class weight should be a list. 
        """
        super().__init__()
        self.device = device
        self.class_weight = torch.tensor(class_weight, device=device)
        self.num_classes = num_classes
    def forward(self, y_true, y_pred):
        yTrueOnehot = torch.zeros(y_true.size(
            0), self.num_classes, device=self.device)
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)

        y_pred = torch.clamp(y_pred, min=1e-6, max=1-1e-6)
        active = yTrueOnehot * (1 - y_pred)
        loss = active * self.class_weight
        return torch.sum(loss) / (torch.sum(self.class_weight) * y_true.size(0))

class SupConLoss(nn.Module):
    def __init__(self, device, temperature=0.1,
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, dim=784...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) != 3:
            raise ValueError("features needs to be [bsz, n_views, ...], \
                3 dimensions are required")

        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(contrast_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = 1 - torch.eye(batch_size * contrast_count, device=self.device)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        return loss.mean()