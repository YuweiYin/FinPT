# -*- coding: utf-8 -*-

import numpy as np
from model.base import ModelSML


class ModelLogisticRegression(ModelSML):

    def __init__(self, args,
                 random_state=None, C=None, solver=None, max_iter=None):
        super().__init__(args)

        from sklearn.linear_model import LogisticRegression

        self.random_state = random_state
        self.C = C
        self.solver = solver
        self.max_iter = max_iter

        self.param_grid = {
            "random_state": [0, 1, 42, 1234],
            "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
            "max_iter": [100, 1000],
        }

        self.model = LogisticRegression(
            penalty="l2",
            dual=False,
            tol=1e-4,
            C=1.0 if self.C is None else self.C,
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=args.class_weight if hasattr(args, "class_weight") else None,
            random_state=args.seed if self.random_state is None else self.random_state,
            solver="lbfgs" if self.solver is None else self.solver,
            max_iter=1000 if self.max_iter is None else self.max_iter,  # original: 100
            multi_class="auto",
            verbose=0,
            warm_start=False,
            n_jobs=-1,  # default: None
            l1_ratio=None,
        )


class ModelPerceptron(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.linear_model import Perceptron

        self.model = Perceptron(
            random_state=args.seed,
            class_weight=args.class_weight if hasattr(args, "class_weight") else None,
            n_jobs=-1,  # default: None
        )


class ModelRidgeClassifier(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.linear_model import RidgeClassifier

        self.model = RidgeClassifier(
            random_state=args.seed,
            class_weight=args.class_weight if hasattr(args, "class_weight") else None,
        )


class ModelLasso(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.linear_model import Lasso

        self.model = Lasso(
            random_state=args.seed,
        )
        # class_weight=args.class_weight if hasattr(args, "class_weight") else None,


class ModelSGDClassifier(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.linear_model import SGDClassifier

        self.model = SGDClassifier(
            random_state=args.seed,
            class_weight=args.class_weight if hasattr(args, "class_weight") else None,
            n_jobs=-1,  # default: None
        )


class ModelBernoulliNB(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.naive_bayes import BernoulliNB

        self.model = BernoulliNB()


class ModelMultinomialNB(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.naive_bayes import MultinomialNB

        self.model = MultinomialNB()


class ModelGaussianNB(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.naive_bayes import GaussianNB

        self.model = GaussianNB()


class ModelDecisionTreeClassifier(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.tree import DecisionTreeClassifier

        self.model = DecisionTreeClassifier(
            random_state=args.seed,
            class_weight=args.class_weight if hasattr(args, "class_weight") else None,
        )


class ModelGradientBoostingClassifier(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.ensemble import GradientBoostingClassifier

        self.model = GradientBoostingClassifier(
            random_state=args.seed,
        )
        # class_weight=args.class_weight if hasattr(args, "class_weight") else None,


class ModelRandomForestClassifier(ModelSML):

    def __init__(self, args,
                 random_state=None, n_estimators=None, max_depth=None):
        super().__init__(args)

        from sklearn.ensemble import RandomForestClassifier

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.param_grid = {
            "random_state": [0, 1, 42, 1234],
            "n_estimators": list(range(80, 200, 10)),
            "max_depth": list(range(2, 15, 1)),
        }

        self.model = RandomForestClassifier(
            n_estimators=100 if self.n_estimators is None else self.n_estimators,
            criterion="gini",
            max_depth=None if self.max_depth is None else self.max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,  # default: None
            random_state=args.seed if self.random_state is None else self.random_state,
            verbose=0,
            warm_start=False,
            class_weight=args.class_weight if hasattr(args, "class_weight") else None,
            ccp_alpha=0.0,
            max_samples=None,
        )


class ModelXGBClassifier(ModelSML):

    def __init__(self, args,
                 random_state=None, n_estimators=None, max_depth=None, learning_rate=None,
                 subsample=None, colsample_bytree=None, min_child_weight=None):
        super().__init__(args)

        from xgboost import XGBClassifier

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight

        self.param_grid = {
            "random_state": [0, 1, 42, 1234],
            "n_estimators": list(range(80, 200, 10)),
            "max_depth": list(range(2, 15, 1)),
            "learning_rate": list(np.linspace(0.01, 2, 20)),
            "subsample": list(np.linspace(0.7, 1.0, 20)),
            "colsample_bytree": list(np.linspace(0.5, 1.0, 10)),
            "min_child_weight": list(range(1, 20, 2)),
        }

        self.model = XGBClassifier(
            max_depth=None if self.max_depth is None else self.max_depth,
            max_leaves=None,
            max_bin=None,
            grow_policy=None,
            learning_rate=None if self.learning_rate is None else self.learning_rate,
            n_estimators=100 if self.n_estimators is None else self.n_estimators,
            verbosity=None,
            objective=None,
            booster=None,
            tree_method=None,
            n_jobs=-1,  # default: None
            gamma=None,
            min_child_weight=None if self.min_child_weight is None else self.min_child_weight,
            max_delta_step=None,
            subsample=None if self.subsample is None else self.subsample,
            sampling_method=None,
            colsample_bytree=None if self.colsample_bytree is None else self.colsample_bytree,
            colsample_bylevel=None,
            colsample_bynode=None,
            reg_alpha=None,
            reg_lambda=None,
            scale_pos_weight=None,
            base_score=None,
            random_state=args.seed if self.random_state is None else self.random_state,
            missing=np.nan,
            num_parallel_tree=None,
            monotone_constraints=None,
            interaction_constraints=None,
            importance_type=None,
            gpu_id=None,
            validate_parameters=None,
            predictor=None,
            enable_categorical=False,
            feature_types=None,
            max_cat_to_onehot=None,
            max_cat_threshold=None,
            eval_metric=None,
            early_stopping_rounds=None,
            callbacks=None,
        )
        # class_weight=args.class_weight if hasattr(args, "class_weight") else None,


class ModelCatBoostClassifier(ModelSML):

    def __init__(self, args,
                 random_state=None, n_estimators=None, max_depth=None, learning_rate=None, subsample=None):
        super().__init__(args)

        from catboost import CatBoostClassifier

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample

        self.param_grid = {
            "random_state": [0, 1, 42, 1234],
            "n_estimators": list(range(80, 200, 10)),
            "max_depth": list(range(2, 15, 1)),
            "learning_rate": list(np.linspace(0.01, 2, 20)),
            "subsample": list(np.linspace(0.7, 1.0, 20)),
        }

        self.model = CatBoostClassifier(
            iterations=None,
            learning_rate=None if self.learning_rate is None else self.learning_rate,
            depth=None,
            l2_leaf_reg=None,
            model_size_reg=None,
            rsm=None,
            loss_function=None,
            border_count=None,
            feature_border_type=None,
            per_float_feature_quantization=None,
            input_borders=None,
            output_borders=None,
            fold_permutation_block=None,
            od_pval=None,
            od_wait=None,
            od_type=None,
            nan_mode=None,
            counter_calc_method=None,
            leaf_estimation_iterations=None,
            leaf_estimation_method=None,
            thread_count=None,
            random_seed=None,
            use_best_model=None,
            best_model_min_trees=None,
            verbose=False,
            silent=None,
            logging_level=None,
            metric_period=None,
            ctr_leaf_count_limit=None,
            store_all_simple_ctr=None,
            max_ctr_complexity=None,
            has_time=None,
            allow_const_label=None,
            target_border=None,
            classes_count=None,
            class_weights=args.class_weight if hasattr(args, "class_weight") else None,
            auto_class_weights=None,
            class_names=None,
            one_hot_max_size=None,
            random_strength=None,
            name=None,
            ignored_features=None,
            train_dir=None,
            custom_loss=None,
            custom_metric=None,
            eval_metric=None,
            bagging_temperature=None,
            save_snapshot=None,
            snapshot_file=None,
            snapshot_interval=None,
            fold_len_multiplier=None,
            used_ram_limit=None,
            gpu_ram_part=None,
            pinned_memory_size=None,
            allow_writing_files=None,
            final_ctr_computation_mode=None,
            approx_on_full_history=None,
            boosting_type=None,
            simple_ctr=None,
            combinations_ctr=None,
            per_feature_ctr=None,
            ctr_description=None,
            ctr_target_border_count=None,
            task_type=None,
            device_config=None,
            devices=None,
            bootstrap_type=None,
            subsample=None if self.subsample is None else self.subsample,
            mvs_reg=None,
            sampling_unit=None,
            sampling_frequency=None,
            dev_score_calc_obj_block_size=None,
            dev_efb_max_buckets=None,
            sparse_features_conflict_fraction=None,
            max_depth=None if self.max_depth is None else self.max_depth,
            n_estimators=None if self.n_estimators is None else self.n_estimators,
            num_boost_round=None,
            num_trees=None,
            colsample_bylevel=None,
            random_state=args.seed if self.random_state is None else self.random_state,
            reg_lambda=None,
            objective=None,
            eta=None,
            max_bin=None,
            scale_pos_weight=None,
            gpu_cat_features_storage=None,
            data_partition=None,
            metadata=None,
            # early_stopping_rounds=None,
            cat_features=None,
            grow_policy=None,
            min_data_in_leaf=None,
            min_child_samples=None,
            max_leaves=None,
            num_leaves=None,
            score_function=None,
            leaf_estimation_backtracking=None,
            ctr_history_unit=None,
            monotone_constraints=None,
            feature_weights=None,
            penalties_coefficient=None,
            first_feature_use_penalties=None,
            per_object_feature_penalties=None,
            model_shrink_rate=None,
            model_shrink_mode=None,
            langevin=None,
            diffusion_temperature=None,
            posterior_sampling=None,
            boost_from_average=None,
            text_features=None,
            tokenizers=None,
            dictionaries=None,
            feature_calcers=None,
            text_processing=None,
            embedding_features=None,
            callback=None,
            eval_fraction=None,
            fixed_binary_splits=None,
        )


class ModelLGBMClassifier(ModelSML):

    def __init__(self, args,
                 random_state=None, n_estimators=None, max_depth=None, learning_rate=None,
                 subsample=None, colsample_bytree=None, min_child_weight=None):
        super().__init__(args)

        from lightgbm import LGBMClassifier

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight

        self.param_grid = {
            "random_state": [0, 1, 42, 1234],
            "n_estimators": list(range(80, 200, 10)),
            "max_depth": list(range(2, 15, 1)),
            "learning_rate": list(np.linspace(0.01, 2, 20)),
            "subsample": list(np.linspace(0.7, 1.0, 20)),
            "colsample_bytree": list(np.linspace(0.5, 1.0, 10)),
            "min_child_weight": list(range(1, 20, 2)),
        }

        self.model = LGBMClassifier(
            boosting_type="gbdt",
            num_leaves=31,
            max_depth=-1 if self.max_depth is None else self.max_depth,
            learning_rate=0.1 if self.learning_rate is None else self.learning_rate,
            n_estimators=100 if self.n_estimators is None else self.n_estimators,
            subsample_for_bin=200000,
            objective=None,
            class_weight=args.class_weight if hasattr(args, "class_weight") else None,
            min_split_gain=0.,
            min_child_weight=1e-3 if self.min_child_weight is None else self.min_child_weight,
            min_child_samples=20,
            subsample=1. if self.subsample is None else self.subsample,
            subsample_freq=0,
            colsample_bytree=1. if self.colsample_bytree is None else self.colsample_bytree,
            reg_alpha=0.,
            reg_lambda=0.,
            random_state=args.seed if self.random_state is None else self.random_state,
            n_jobs=-1,
            silent='warn',
            importance_type='split',
        )
