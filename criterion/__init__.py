from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, CTCLoss, NLLLoss, PoissonNLLLoss, GaussianNLLLoss, \
    KLDivLoss, BCELoss, BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, \
    HuberLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, \
    TripletMarginLoss, TripletMarginWithDistanceLoss

CRITERION_DICT = {
    "L1Loss": L1Loss, "MSELoss": MSELoss,  "CrossEntropyLoss": CrossEntropyLoss, "CTCLoss": CTCLoss,
    "NLLLoss": NLLLoss, "PoissonNLLLoss": PoissonNLLLoss, "GaussianNLLLoss": GaussianNLLLoss,
    "KLDivLoss": KLDivLoss, "BCELoss": BCELoss, "BCEWithLogitsLoss": BCEWithLogitsLoss,
    "MarginRankingLoss": MarginRankingLoss, "HingeEmbeddingLoss": HingeEmbeddingLoss,
    "MultiLabelMarginLoss": MultiLabelMarginLoss, "HuberLoss": HuberLoss, "SmoothL1Loss": SmoothL1Loss,
    "SoftMarginLoss": SoftMarginLoss, "MultiLabelSoftMarginLoss": MultiLabelSoftMarginLoss,
    "CosineEmbeddingLoss": CosineEmbeddingLoss, "MultiMarginLoss": MultiMarginLoss,
    "TripletMarginLoss": TripletMarginLoss, "TripletMarginWithDistanceLoss": TripletMarginWithDistanceLoss
}

__all__ = [
    "CRITERION_DICT",
    "L1Loss", "MSELoss",  "CrossEntropyLoss", "CTCLoss", "NLLLoss", "PoissonNLLLoss", "GaussianNLLLoss",
    "KLDivLoss", "BCELoss", "BCEWithLogitsLoss", "MarginRankingLoss", "HingeEmbeddingLoss", "MultiLabelMarginLoss",
    "HuberLoss", "SmoothL1Loss", "SoftMarginLoss", "MultiLabelSoftMarginLoss", "CosineEmbeddingLoss",
    "MultiMarginLoss", "TripletMarginLoss", "TripletMarginWithDistanceLoss"
]

classes = __all__
