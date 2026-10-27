"""
Pipeline module for tabular diabetes classification in nfhs-diabetes-ml.

Defines feature sets, ColumnTransformer preprocessing (scaling numeric features,
one-hot encoding categorical attributes), SMOTE oversampling, and model pipelines
for Logistic Regression, Random Forest, LightGBM, and XGBoost.
"""

from typing import Dict, List, Optional, Tuple, Any
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb

from src.config import RANDOM_STATE
from src.utils import setup_logger

logger = setup_logger(__name__)

# 1. Feature Specifications
NUMERIC_FEATURES: List[str] = [
    "age",
    "bmi",
    "height_cm",
    "weight_kg",
    "systolic_bp",
    "diastolic_bp",
    "s706a",  # Milk/curd frequency (1: Daily - 4: Never)
    "s706b",  # Pulses/beans frequency
    "s706c",  # Green leafy vegetables frequency
    "s706d",  # Fruits frequency
    "s706e",  # Eggs frequency
    "s706f",  # Fish frequency
    "s706g",  # Chicken/meat frequency
    "s706h",  # Fried food frequency
    "s706i",  # Aerated drinks frequency
]

BINARY_FEATURES: List[str] = [
    "s728c",  # Doctor diagnosed hypertension (0 = No, 1 = Yes)
    "s728d",  # Doctor diagnosed cancer
    "s728e",  # Doctor diagnosed asthma
    "s728f",  # Doctor diagnosed thyroid disorder
    "s728g",  # Doctor diagnosed heart disease
    "v463a",  # Smokes cigarettes/tobacco
    "s716",   # Drinks alcohol
]

CATEGORICAL_FEATURES: List[str] = [
    "residence",        # Urban, Rural
    "education",        # No education, Primary, Secondary, Higher
    "wealth_group",     # Poorest, Poorer, Middle, Richer, Richest
    "marital_status",   # Never married, Currently married, Formerly married
    "region",           # North, Central, East, Northeast, West, South
    "occupation",       # Not working, Currently working
    "health_insurance", # Not covered, Covered
    "bmi_category",     # Underweight (<18.5), Normal (18.5-24.9), Overweight (25.0-29.9), Obese (>=30.0)
]

ALL_FEATURES: List[str] = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Construct ColumnTransformer for numerical scaling and categorical one-hot encoding.

    Returns
    -------
    ColumnTransformer
        Scikit-learn ColumnTransformer instance.
    """
    numeric_transformer = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    binary_transformer = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    categorical_transformer = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("bin", binary_transformer, BINARY_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_model_catalog(
    scale_pos_weight: float = 43.56,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Tuple[Any, bool]]:
    """
    Retrieve configured candidate estimators paired with their imbalance strategy flag.

    Parameters
    ----------
    scale_pos_weight : float, default=43.56
        Negative to positive class ratio for XGBoost/LightGBM.
    random_state : int, default=RANDOM_STATE (42)
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Tuple[Any, bool]]
        Dictionary mapping model key -> (estimator_instance, use_smote_flag).
    """
    catalog = {
        # 1. Logistic Regression
        "Logistic Regression (Class-Weighted)": (
            LogisticRegression(
                class_weight="balanced",
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=random_state,
            ),
            False,
        ),
        "Logistic Regression (SMOTE)": (
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=random_state,
            ),
            True,
        ),

        # 2. Random Forest
        "Random Forest (Class-Weighted)": (
            RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            False,
        ),
        "Random Forest (SMOTE)": (
            RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=random_state,
                n_jobs=-1,
            ),
            True,
        ),

        # 3. LightGBM
        "LightGBM (Class-Weighted)": (
            lgb.LGBMClassifier(
                class_weight="balanced",
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1,
            ),
            False,
        ),
        "LightGBM (SMOTE)": (
            lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1,
            ),
            True,
        ),

        # 4. XGBoost
        "XGBoost (Scale-Pos-Weight)": (
            xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
            False,
        ),
        "XGBoost (SMOTE)": (
            xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
            True,
        ),
    }

    return catalog


def build_model_pipeline(
    model: Any,
    preprocessor: Optional[ColumnTransformer] = None,
    use_smote: bool = False,
    random_state: int = RANDOM_STATE,
) -> ImbPipeline:
    """
    Construct a full end-to-end classification pipeline with optional SMOTE.

    Parameters
    ----------
    model : Any
        Scikit-learn compatible classifier.
    preprocessor : ColumnTransformer, optional
        Preprocessing ColumnTransformer. If None, built via `build_preprocessor()`.
    use_smote : bool, default=False
        Whether to inject SMOTE oversampling into the pipeline training phase.
    random_state : int, default=RANDOM_STATE
        Random state for SMOTE.

    Returns
    -------
    ImbPipeline
        Imbalanced-learn Pipeline instance.
    """
    transformer = preprocessor if preprocessor is not None else build_preprocessor()

    steps = [("preprocessor", transformer)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("classifier", model))

    return ImbPipeline(steps=steps)
