import json
import logging

logger = logging.getLogger(__name__)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            json.dump(data, f, indent=4, sort_keys=sort_keys)
        else:
            json.dump(data, f)


def save_parsed_args_to_json(parsed_args, file_path, pretty=True):
    save_json(vars(parsed_args), file_path, save_pretty=pretty)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def set_lr(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * decay_factor


def flat_list_of_lists(l):
    """Flatten [[1, 2], [3, 4]] → [1, 2, 3, 4]."""
    return [item for sublist in l for item in sublist]


def count_parameters(model, verbose=True):
    """Count total and trainable parameters in a PyTorch model."""
    n_all        = sum(p.numel() for p in model.parameters())
    n_trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        logger.info(
            "Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable)
        )
    return n_all, n_trainable


def sum_parameters(model, verbose=True):
    """Sum all parameter values (useful as a quick sanity check)."""
    p_sum = sum(p.sum().item() for p in model.parameters())
    if verbose:
        logger.info("Parameter sum %s", p_sum)
    return p_sum


def merge_dicts(list_dicts):
    """Merge a list of dicts left-to-right (later keys win)."""
    merged = list_dicts[0].copy()
    for d in list_dicts[1:]:
        merged.update(d)
    return merged


def merge_json_files(paths, merged_path):
    save_json(merge_dicts([load_json(p) for p in paths]), merged_path)