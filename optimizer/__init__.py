from torch.optim import Adadelta, Adagrad, Adam, AdamW, Adamax, SparseAdam, ASGD, SGD, LBFGS, RMSprop, RAdam, Rprop
from .lion_pytorch import Lion

OPTIMIZER_DICT = {
    "Adadelta": Adadelta, "Adagrad": Adagrad, "Adam": Adam, "AdamW": AdamW, "Adamax": Adamax, "SparseAdam": SparseAdam,
    "ASGD": ASGD, "SGD": SGD, "LBFGS": LBFGS, "RMSprop": RMSprop, "RAdam": RAdam, "Rprop": Rprop,
    "Lion": Lion,
}

__all__ = [
    "OPTIMIZER_DICT",
    "Adadelta", "Adagrad", "Adam", "AdamW", "Adamax", "SparseAdam",
    "ASGD", "SGD", "LBFGS", "RMSprop", "RAdam", "Rprop",
    "Lion",
]

classes = __all__
