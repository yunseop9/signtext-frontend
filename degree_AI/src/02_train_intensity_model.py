import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_NAMES = ["weak", "normal", "strong"]


def train_model(data_dir: Path, model_dir: Path, mode: str, model_type: str) -> None:
    X_path = data_dir / f"X_{mode}.npy"
    y_path = data_dir / f"y_{mode}.npy"
    meta_path = data_dir / f"meta_{mode}.csv"

    if not X_path.exists():
        raise FileNotFoundError(f"Missing: {X_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Missing: {y_path}")

    X = np.load(X_path)
    y = np.load(y_path)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    label_counts = pd.Series(y).value_counts().sort_index()
    print("\nLabel counts:")
    for idx, count in label_counts.items():
        print(f"  {idx} ({LABEL_NAMES[int(idx)]}): {count}")

    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        if "subject" in meta.columns:
            groups = meta["subject"].values

            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=0.2,
                random_state=42
            )

            train_idx, val_idx = next(splitter.split(X, y, groups=groups))
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

    elif model_type == "mlp":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=512,
                learning_rate_init=1e-3,
                max_iter=80,
                random_state=42,
                verbose=True
            ))
        ])

    else:
        raise ValueError("model_type must be 'rf' or 'mlp'")

    print(f"\nTraining model: {model_type}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print("\nClassification Report")
    print(classification_report(
        y_val,
        y_pred,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        zero_division=0
    ))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_val, y_pred, labels=[0, 1, 2]))

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"degree_{mode}_{model_type}.joblib"

    bundle = {
        "model": model,
        "mode": mode,
        "model_type": model_type,
        "label_names": LABEL_NAMES,
        "feature_type": "common_face_degree_features"
    }

    joblib.dump(bundle, model_path)

    print(f"\nSaved model: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--mode", type=str, default="anger", choices=["anger", "overall"])
    parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "mlp"])
    args = parser.parse_args()

    train_model(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        mode=args.mode,
        model_type=args.model_type
    )


if __name__ == "__main__":
    main()