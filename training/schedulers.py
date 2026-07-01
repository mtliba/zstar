import math


def linear_kl_anneal(epoch: int, warmup_epochs: int, target_beta: float) -> float:
    if warmup_epochs <= 0:
        return target_beta
    return min(target_beta, target_beta * epoch / warmup_epochs)


def cyclical_kl_anneal(epoch: int, cycle_length: int, target_beta: float, ratio: float = 0.5) -> float:
    tau = (epoch % cycle_length) / cycle_length
    if tau <= ratio:
        return target_beta * tau / ratio
    return target_beta


def get_kl_beta(epoch: int, cfg) -> float:
    schedule = str(cfg.training.get("kl_schedule", "linear"))
    warmup = int(cfg.training.get("kl_warmup_epochs", 20))
    target = float(cfg.model.get("beta", 4.0))

    if schedule == "linear":
        return linear_kl_anneal(epoch, warmup, target)
    elif schedule == "cyclical":
        return cyclical_kl_anneal(epoch, warmup, target)
    elif schedule == "monotonic":
        return target
    raise ValueError(f"Unknown KL schedule '{schedule}'")
