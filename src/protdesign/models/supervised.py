"""
Supervised regression models trained on top of embeddings and/or scores from zero-shot models
"""
from typing import Any, Sequence, Literal
import numpy as np
from protdesign.dataset import LabeledInstanceDataset, LabeledInstanceTrainTestDataset
from protdesign.entity import System, SystemInstance
from protdesign.model import Transformer, Scorer, RequiredResources, SupervisedBaseModel, MutationScorer, \
    ConditionalMutationScorer
from protdesign.types import StatusCallback, ModelStats
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate, cross_val_predict, KFold
from sklearn.utils import all_estimators
from sklearn.utils.validation import check_is_fitted
from scipy.stats import pearsonr, spearmanr

r2_scorer = make_scorer(r2_score)

spearman_scorer = make_scorer(
    lambda y_pred, y_true: spearmanr(y_pred, y_true).correlation
)

pearson_scorer = make_scorer(
    lambda y_pred, y_true: pearsonr(y_pred, y_true).correlation
)


class SklearnRegressorOnEmbeddings(SupervisedBaseModel, Scorer, MutationScorer, ConditionalMutationScorer):
    """
    Supervised property prediction from pooled molecular embeddings. Can stack any
    scikit-learn-compatible predictors that implement fit() and predict()
    methods, including pipelines

    Note that passed in data must be pre-transformed (higher values for more functional/fit
    sequences and lower values for less functional/fit sequences), ideally on a log-like scale;
    e.g. log-transformed read ratios vs WT

    High-priority future feature additions:
    - Handle multi-entity systems (need to implement strategy for combining different
      entity embeddings, either by pooling or stacking; need to handle potentially different feature
      vector dimensions)
    - Implement non-random splitting strategies to get more meaningful estimates of performance
    - Multi-system learning from data

    Lower-priority future feature additions:
    - Generalize to also allow classification (only minor modifications needed to scoring on test set)
    - Explicit embedding normalization (e.g. relative to reference sequence)
    - Multi-label learning (but need to think about how to integrate with score() with expects scalar per instance)
    - Allow to specify separate model for density feature
    """
    available = True
    name: str = "Supervised predictor on sequence embeddings"
    citation_strings: list[str] = ["unpublished"]  # TODO: update

    # core properties
    requires_target: bool = False
    requires_fixed_length: bool = False
    handles_deletions: bool = True
    handles_insertions: bool = True
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = True

    # molecular model properties
    requires_heavy_build: bool = False
    requires_seqs: bool = False
    requires_msa: bool = False
    requires_3d: bool = False

    def __init__(
        self,
        embedder: Transformer | None,
        predictor: Any | str,
        predictor_kwargs: dict[str, Any] | None = None,
        target_name: str | None = None,
        override_embedder_for_training: bool = False,
        use_scores: bool = True,
        use_embeddings: bool = True,
        pooling: Literal["mean", "max"] = "mean",
        cv_folds: int | None = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Train supervised regression model on top of molecular model embeddings/scores. Positional embeddings
        will be pooled to one feature vector along the position dimension.

        Can be used in either of two modes with pre-computed embeddings/scores, or through on-the-fly computation
        (cf. embedder param). The latter mode is needed to use mutation scoring methods, e.g. for Gibbs sampling
        or calculation of single mutation matrices.

        Parameters
        ----------
        embedder
            Molecular model to use for computing embeddings/scores on the fly
            (if None, will use values available on supplied instances for build() and score(); in this mode,
            mutation scoring methods cannot be used). Also note override_embedder_for_training for multi-system
            training of models.
        predictor
            Scikit-learn regressor instance or model name string as available through
            sklearn.utils.all_estimators(type_filter="regressor")
        predictor_kwargs
            Constructor parameters to use if predictor is a string (will be ignored if predictor is a model instance)
        target_name
            Name of target series in LabeledInstanceDataset to retrieve. If the dataset only contains a single series,
            it can be extracted as a default by setting this parameter to None (an exception will be raised otherwise)
        override_embedder_for_training
            If True, use embeddings/score on instances, even if embedder is specified. This allows to train
            a model on a dataset with instances from multiple systems (e.g. stability measurements for many different
            proteins). The embedder will still be used at prediction time to allow mutation prediction methods
            to be used.
        use_scores
            If True, include instance score as a model feature (will raise an error if scores are absent and
            cannot be computed with embedder)
        use_embeddings
            IF True, include embeddings as a model feature (will raise an error if embeddings are absent and
            cannot be computed with embedder
        pooling
            Aggregation to apply to positional embeddings across position dimension (to obtain one feature vector
            per entity)
        cv_folds
            Number of cross-validation folds to use during model training, if no explicit test dataset is supplied
            to build()
        random_state
            Number to initialize random state of CV fold splitting (note: will not be applied to predictor, this
            needs to be done during instance construction or using predictor_kwargs if predictor string is supplied)
        n_jobs
            Number of cores to use for scikit-learn computations (-1: use all available cores)
        """
        # instantiate predictor from model name string or store provided instance
        if isinstance(predictor, str):
            all_predictors = dict(all_estimators(type_filter="regressor"))
            if predictor in all_predictors:
                if predictor_kwargs is None:
                    predictor_kwargs = {}
                self.predictor: Any = all_predictors[predictor](
                    **predictor_kwargs
                )
            else:
                raise ValueError(
                    f"Invalid regressor, valid options are {', ' .join(list(all_predictors))}"
                )
        else:
            # unfortunately no good typing options available, so verify attributes like scikit-learn does
            if not hasattr(predictor, "fit") or not hasattr(predictor, "predict"):
                raise ValueError(
                    "Predictor must have scikit-learn fit() and predict methods()"
                )

            self.predictor = predictor

        # make sure we are left with some features
        if not use_scores and not use_embeddings:
            raise ValueError(
                "At least one of use_scores or use_embeddings must be True"
            )

        # modelled system
        self._system = None

        # note: embedder needs to be built already built outside by convention if a BaseModel
        self.embedder = embedder
        self.override_embedder_for_training = override_embedder_for_training
        self.target_name = target_name
        self.use_scores = use_scores
        self.use_embeddings = use_embeddings
        self.pooling_strategy = pooling
        self.predictor_kwargs = predictor_kwargs if predictor_kwargs is not None else {}
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs

        # update class variables on instance as these will be used by mixin scoring function defaults
        if self.embedder is not None:
            self.handles_insertions = embedder.handles_insertions
            self.handles_deletions = embedder.handles_deletions
            self.requires_fixed_length = embedder.requires_fixed_length
            self.requires_target = embedder.requires_target

        # performance statistics
        self._y_true = None
        self._y_pred = None
        self._spearman = None
        self._pearson = None
        self._r2 = None

    @property
    def ready(self):
        # model only required if embeddings are non pre-specified
        fitted = True
        try:
            check_is_fitted(self.predictor)
        except NotFittedError:
            fitted = False

        return self.system is not None and fitted

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def required_resources(
        cls, system: System, data: Any, use_gpu: bool = True,
        build: bool = True
    ) -> RequiredResources:
        raise NotImplementedError(
            "Resource estimation not yet implemented"
        )

    def positions(
        self,
        instance: SystemInstance | None = None,
    ) -> list[tuple[int, int]]:
        self.ready_or_raise()

        if self.embedder is not None:
            return self.embedder.positions(instance)
        else:
            raise ValueError(
                "No explicit embedder specified, cannot use positions()"
            )

    @classmethod
    def can_model(cls, system: System, data: LabeledInstanceDataset) -> tuple[bool, str]:
        if len(system) != 1:
            return False, "Can currently only handle single-component systems"

        if data is None:
            return False, "Labelled instance must be supplied for building model"

        return True, ""

    def _transform_and_validate_instances(
        self,
        instances: Sequence[SystemInstance],
        override_embedder: bool,
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int, int], np.dtype[float]]:
        # start with instances as they are as transformed instances, will add to these as needed further down
        instances_t = instances

        if self.use_embeddings:
            # compute embeddings on the fly and replace instances if we have the model explicitly specified;
            # pass status_callback through as this is mostly heavy part of the computation
            if self.embedder is not None and not override_embedder:
                # in this case, we leave instance validation to the embedder
                instances_t = self.embedder.transform(
                    instances, entity=None, status_callback=status_callback
                )
            else:
                # perform instance validation; this does not imply all instances actually have an embedding
                # so must check this as well
                [
                    self.system.valid_instance(
                        instance,
                        validate_reps=True,
                        require_reps=False,
                        validate_embeddings=True,
                        fixed_length=False,
                        allow_deletions=True,
                        raise_invalid=True,
                    ) for instance in instances_t
                ]

            # extract embeddings and verify they are complete; implementation right now
            # assumes single entity case... need to define a strategy for assembling multiple
            # entities either by pooling or stacking pooled vector;
            # note: not creating a numpy array on outer dimension as length of embeddings in position
            # dimension may vary
            embeddings = [
                inst[0].embedding for inst in instances_t if inst[0].embedding is not None
            ]

            # check embedding completeness
            if len(embeddings) != len(instances_t):
                raise ValueError(
                    "All instances must have an embedding if use_embeddings is True. "
                    "Precompute or specify a model to compute on the fly."
                )

            # check embeddings all have same dimensionality (vector or matrix)
            embedding_shapes = {
                len(emb.shape) for emb in embeddings
            }
            if len(embedding_shapes) != 1:
                raise ValueError(
                    f"Embeddings must all have same shape (vector or matrix) but found {embedding_shapes}"
                )

            embedding_dims = {
                emb.shape[-1] for emb in embeddings
            }
            if len(embedding_dims) != 1:
                raise ValueError(
                    f"Embeddings must all have same feature dimensionality but found {embedding_dims}"
                )

            # if embedding matrix, apply pooling across sequence dimension;
            # use nan versions of functions to allow blanking out other positions
            if list(embedding_shapes)[0] == 2:
                if self.pooling_strategy == "mean":
                    pooling_func = np.nanmean
                elif self.pooling_strategy == "max":
                    pooling_func = np.nanmax
                else:
                    raise ValueError("Invalid pooling strategy")

                embeddings = np.array(
                    [pooling_func(emb, axis=0) for emb in embeddings]
                )
        else:
            embeddings = np.zeros((len(instances_t), 0))

        if self.use_scores:
            # extract scores from transformed instances (if using transform() above, these may have been
            # computed already)
            scores = np.array([
                inst.score for inst in instances_t if inst.score is not None
            ])

            # handle missing scores
            if len(scores) != len(instances_t):
                # if we have an embedder, we can recompute
                if self.embedder is not None and isinstance(self.embedder, Scorer) and not override_embedder:
                    scores = self.embedder.score(instances)
                else:
                    raise ValueError(
                        "All instances must have a score if use_score is True. Precompute or specify a model "
                        "to compute on the fly."
                    )
            scores = scores[:, np.newaxis]
        else:
            scores = np.zeros((len(instances_t), 0))

        # concatenate along feature dimension and return
        x = np.concatenate(
            (embeddings, scores), axis=1
        )

        return x

    def build(
        self,
        system: System,
        data: LabeledInstanceTrainTestDataset,
        status_callback: StatusCallback | None = None
    ):
        # verify if we can model the system
        self.can_model_or_raise(system, data)

        # make record of modelled system
        self._system = system

        if self.embedder is not None and self.system != self.embedder.system:
            raise ValueError(
                "system does not agree to embedder"
            )

        if ((isinstance(self.predictor, GridSearchCV) or isinstance(self.predictor, RandomizedSearchCV)) and
                data.test_set is None):
            raise ValueError(
                "Must specify explicit test set for crossvalidation-based parameter search methods"
            )

        # retrieve target series, do not use missing values
        train_instances, train_values = data.training_set.select(
            self.target_name, drop_missing=True
        )

        # training set
        x_train = self._transform_and_validate_instances(
            train_instances, self.override_embedder_for_training, status_callback
        )
        y_train = np.array(train_values)

        # explicitly specified test set, if available, do not use cross-validation for performance estimation
        if data.test_set is not None:
            test_instances, test_values = data.test_set.select(
                self.target_name, drop_missing=True
            )

            x_test = self._transform_and_validate_instances(
                test_instances, self.override_embedder_for_training, status_callback
            )
            y_test = np.array(test_values)
        else:
            x_test = None
            y_test = None

        if x_test is None:
            # estimate performance with cross validation

            # shuffle dataset, default flr cross_validate is shuffle=False
            k_fold = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )

            cv_results = cross_validate(
                self.predictor,
                x_train,
                y_train,
                scoring={
                    "spearman": spearman_scorer,
                    "pearson": pearson_scorer,
                    "r2": r2_scorer,
                },
                cv=k_fold,
                n_jobs=self.n_jobs
            )
            self._spearman = cv_results["test_spearman"] #.mean()
            self._pearson = cv_results["test_pearson"] #.mean()
            self._r2 = cv_results["test_r2"] #.mean()

            # create predicted values with cross-validation
            self._y_pred = cross_val_predict(
                self.predictor,
                x_train,
                y_train,
                cv=k_fold,
                n_jobs=self.n_jobs
            )
            self._y_true = y_train

            # refit final predictor on whole dataset
            self.predictor.fit(x_train, y_train)
        else:
            # fit final predictor on full training set (this could also implicitly be GridSearchCV/RandomSearchCV)
            self.predictor.fit(x_train, y_train)

            # evaluate on test set
            self._y_pred = self.predictor.predict(x_test)
            self._y_true = y_test
            self._spearman = [spearmanr(self._y_pred, self._y_true).correlation] # noqa
            self._pearson = [pearsonr(self._y_pred, self._y_true).correlation]
            self._r2 = [r2_score(self._y_true, self._y_pred)]

        return self

    def stats(self) -> ModelStats | None:
        """
        Return summary statistics about built model (e.g. cross validation statistics) after
        a model has been prepared with build()

        Returns
        -------
        Model statistics
        """
        # only able to provide statistics once model has been built
        self.ready_or_raise()

        return {
            "y_true": self._y_true,
            "y_pred": self._y_pred,
            "spearman": self._spearman,
            "pearson": self._pearson,
            "r2": self._r2,
        }

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None
    ) -> np.ndarray[tuple[int], np.dtype[float]]:
        self.ready_or_raise()

        x_pred = self._transform_and_validate_instances(
            instances, override_embedder=False, status_callback=status_callback
        )

        return self.predictor.predict(
            x_pred
        )
