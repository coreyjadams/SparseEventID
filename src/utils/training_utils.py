import torch
import datetime

import torch_optimizer as optim

def init_optimizer(optimizer_kind, parameters, decay):

    from src.config.optimizer import OptimizerKind

    if optimizer_kind == OptimizerKind.rmsprop:
        opt = torch.optim.RMSprop(parameters, lr=1.0, eps=1e-6, weight_decay=decay)
    elif optimizer_kind == OptimizerKind.adam:
        opt = torch.optim.Adam(parameters, lr=1.0, eps=1e-6, betas=(0.8,0.9), weight_decay=decay)
    elif optimizer_kind == OptimizerKind.adagrad:
        opt = torch.optim.Adagrad(parameters, lr=1.0, weight_decay=decay)
    elif optimizer_kind == OptimizerKind.adadelta:
        opt = torch.optim.Adadelta(parameters, lr=1.0, eps=1e-6, weight_decay=decay)
    elif optimizer_kind == OptimizerKind.lars:
        from . lars import LARS
        opt = LARS(parameters, lr=1.0, weight_decay=decay)
    elif optimizer_kind == OptimizerKind.lamb:
        opt = optim.Lamb(parameters, lr=1.0, weight_decay=decay)
    elif optimizer_kind == OptimizerKind.novograd:
        opt = optim.NovoGrad(parameters, lr=1.0, weight_decay=decay)
    else:
        opt = torch.optim.SGD(parameters, lr=1.0, weight_decay=decay)

    return opt


def format_log_message(log_keys, metrics, batch_size, global_step, mode=""):

    format_log_message.current_log_time = datetime.datetime.now()

    # Build up a string for logging:
    s = ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in metrics])

    time_string = []

    if format_log_message.previous_log_time is not None:
        # try:
        total_images = batch_size
        images_per_second = total_images / (format_log_message.current_log_time -  format_log_message.previous_log_time).total_seconds()
        time_string.append("{:.3} Img/s".format(images_per_second))

    if 'io_fetch_time' in metrics.keys():
        time_string.append("{:.2} IOs".format(metrics['io_fetch_time']))

    if 'step_time' in metrics.keys():
        time_string.append("{:.2} (Step)(s)".format(metrics['step_time']))

    if len(time_string) > 0:
        s += " (" + " / ".join(time_string) + ")"

    format_log_message.previous_log_time = datetime.datetime.now()

    return "{} Step {} metrics: {}".format(mode, global_step, s)

format_log_message.previous_log_time = None
format_log_message.current_log_time  = None
