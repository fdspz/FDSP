from utils.util import get_y
from utils.losses import MMDLoss


def compute_loss(model, model_name, mode, x_train, y_train, domain_label, mse_criterion, label_criterion, learning_type, mmd_weight, dan_weight, device):

    hidden_state = model.encode(x_train, device)
    delta_g = model.predict(hidden_state)
    y_hat = get_y(delta_g, x_train, learning_type, device)
    loss = mse_criterion(y_hat, y_train)
    pred_loss = loss
    belief_loss = -1

    feas = hidden_state
    source_feas = feas[domain_label == 0]
    target_feas = feas[domain_label == 1]


    if mode in ['finetune', 'dir', 'mixed']:
        return loss, [pred_loss, belief_loss, -1, -1]
    elif mode == 'mmd':
        mmd_loss = MMDLoss()(source_feas, target_feas)
        loss = loss + mmd_weight * mmd_loss
        return loss, [pred_loss, belief_loss, -1, mmd_loss]
    elif mode == 'dan':
        label_hat = model.domain_classification(feas)
        label_loss = label_criterion(label_hat, domain_label)
        loss = loss - dan_weight * label_loss   # loss = loss + dan_weight * label_loss vis feas
        return loss, [pred_loss, belief_loss, label_loss, -1]
    elif mode == 'combined':
        label_hat = model.domain_classification(feas)
        label_loss = label_criterion(label_hat, domain_label)

        mmd_loss = MMDLoss()(source_feas, target_feas)
        loss = loss + dan_weight * label_loss + mmd_weight * mmd_loss
        return loss, [pred_loss, belief_loss, label_loss, mmd_loss]
    else:
        raise ValueError(f"Unknown mode: {mode}")