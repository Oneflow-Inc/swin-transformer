import torch

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def load_from_torch(model, torch_dict):
    parameters = torch_dict
    new_parameters = dict()
    for key, value in parameters.items():
        if "num_batches_tracked" not in key:
          val = value.detach().cpu().numpy()
          new_parameters[key] = val
    model.load_state_dict(new_parameters)
    return model