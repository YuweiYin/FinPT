from .base import ModelSML, ModelNN, ModelGNN
from .sml import ModelLogisticRegression, ModelPerceptron, ModelRidgeClassifier, ModelLasso, ModelSGDClassifier, \
    ModelBernoulliNB, ModelMultinomialNB, ModelGaussianNB, ModelDecisionTreeClassifier, \
    ModelGradientBoostingClassifier, ModelRandomForestClassifier, ModelXGBClassifier, \
    ModelCatBoostClassifier, ModelLGBMClassifier
from .deepfm import DeepFM
from .stg import STG
from .vime import VIME
from .tabnet import TabNet


MODEL_DICT = {
    "ModelSML": ModelSML, "SML": ModelSML,
    "ModelNN": ModelNN, "NN": ModelNN,
    "ModelGNN": ModelGNN, "GNN": ModelGNN,
    "ModelLogisticRegression": ModelLogisticRegression, "LogisticRegression": ModelLogisticRegression,
    "ModelPerceptron": ModelPerceptron, "Perceptron": ModelPerceptron,
    "ModelRidgeClassifier": ModelRidgeClassifier, "RidgeClassifier": ModelRidgeClassifier,
    "ModelLasso": ModelLasso, "Lasso": ModelLasso,
    "ModelSGDClassifier": ModelSGDClassifier, "SGDClassifier": ModelSGDClassifier,
    "ModelBernoulliNB": ModelBernoulliNB, "BernoulliNB": ModelBernoulliNB,
    "ModelMultinomialNB": ModelMultinomialNB, "MultinomialNB": ModelMultinomialNB,
    "ModelGaussianNB": ModelGaussianNB, "GaussianNB": ModelGaussianNB,
    "ModelDecisionTreeClassifier": ModelDecisionTreeClassifier, "DecisionTreeClassifier": ModelDecisionTreeClassifier,
    "ModelGradientBoostingClassifier": ModelGradientBoostingClassifier,
    "GradientBoostingClassifier": ModelGradientBoostingClassifier,
    "ModelRandomForestClassifier": ModelRandomForestClassifier, "RandomForestClassifier": ModelRandomForestClassifier,
    "ModelXGBClassifier": ModelXGBClassifier, "XGBClassifier": ModelXGBClassifier,
    "ModelCatBoostClassifier": ModelCatBoostClassifier, "CatBoostClassifier": ModelCatBoostClassifier,
    "ModelLGBMClassifier": ModelLGBMClassifier, "LGBMClassifier": ModelLGBMClassifier,
    "DeepFM": DeepFM, "STG": STG, "VIME": VIME, "TabNet": TabNet,
}

__all__ = [
    "MODEL_DICT",
    "ModelSML", "ModelNN", "ModelGNN",
    "ModelLogisticRegression", "ModelPerceptron", "ModelRidgeClassifier", "ModelLasso", "ModelSGDClassifier",
    "ModelBernoulliNB", "ModelMultinomialNB", "ModelGaussianNB",
    "ModelDecisionTreeClassifier", "ModelGradientBoostingClassifier",
    "ModelRandomForestClassifier", "ModelXGBClassifier", "ModelCatBoostClassifier", "ModelLGBMClassifier",
    "DeepFM", "STG", "VIME", "TabNet",
]

classes = __all__
