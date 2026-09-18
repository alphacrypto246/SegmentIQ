import json
import os
import sys

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split

from src.config import ModelTrainerConfig, N_CLUSTERS_RANGE, RANDOM_STATE
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def select_k(self, features: pd.DataFrame) -> int:
        """
        No ground-truth labels exist for clustering, so k can't be chosen
        by accuracy. Instead we fit KMeans for each candidate k and pick
        the one with the best silhouette score. Davies-Bouldin is logged
        alongside it since relying on a single unsupervised metric is a
        common blind spot — worth naming that tradeoff in an interview.
        """
        try:
            best_k, best_score = None, -1
            for k in N_CLUSTERS_RANGE:
                model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
                labels = model.fit_predict(features)
                sil = silhouette_score(features, labels)
                dbi = davies_bouldin_score(features, labels)
                logging.info(f"k={k}: silhouette={sil:.4f}, davies_bouldin={dbi:.4f}")
                if sil > best_score:
                    best_k, best_score = k, sil

            logging.info(f"Selected k={best_k} (silhouette={best_score:.4f})")
            return best_k
        except Exception as e:
            raise CustomException(e, sys)

    def profile_clusters(self, features: pd.DataFrame, labels) -> dict:
        """
        Translates cluster IDs into plain-English descriptions.

        IMPORTANT: `features` here are already StandardScaler-transformed
        (z-scores, population mean ~0). That means we threshold directly
        on the z-score instead of computing a ratio against the mean —
        dividing by an ~0 mean would produce meaningless, unstable ratios.
        A z-score of +0.5 already means "half a std dev above average."
        """
        try:
            df = features.copy()
            df["cluster"] = labels

            profiles = {}
            for cluster_id, group in df.groupby("cluster"):
                cluster_mean = group.drop(columns=["cluster"]).mean(numeric_only=True)

                high_features = cluster_mean[cluster_mean > 0.5].index.tolist()
                low_features = cluster_mean[cluster_mean < -0.5].index.tolist()

                descriptors = []
                if "ONEOFF_PURCHASES" in high_features or "INSTALLMENTS_PURCHASES" in high_features:
                    descriptors.append("high spenders")
                if "CASH_ADVANCE" in high_features or "CASH_ADVANCE_FREQUENCY" in high_features:
                    descriptors.append("frequent cash-advance users")
                if "PRC_FULL_PAYMENT" in low_features:
                    descriptors.append("low payment ratio (risk segment)")
                if "PRC_FULL_PAYMENT" in high_features:
                    descriptors.append("reliably pays in full")
                if "CREDIT_LIMIT" in high_features and "ONEOFF_PURCHASES" in low_features:
                    descriptors.append("high limit, low activity (upsell target)")
                if "BALANCE" in low_features and "BALANCE_FREQUENCY" in low_features:
                    descriptors.append("low activity / dormant")
                if not descriptors:
                    descriptors.append("average / mixed behavior")

                profiles[str(cluster_id)] = {
                    "size": int(len(group)),
                    "description": ", ".join(descriptors),
                    "mean_features": cluster_mean.round(2).to_dict(),
                }
            return profiles
        except Exception as e:
            raise CustomException(e, sys)

    def train_segment_classifier(self, features: pd.DataFrame, labels):
        """
        Trains the model that actually serves predictions in production.
        This is the fast-inference trick: instead of re-running KMeans or
        computing distances to centroids on every API request, we train a
        supervised classifier to reproduce the cluster assignments. Any
        new customer just gets a single, sub-millisecond .predict() call.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
            )
            clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
            clf.fit(X_train, y_train)
            test_accuracy = clf.score(X_test, y_test)
            logging.info(f"Segment classifier held-out accuracy vs cluster labels: {test_accuracy:.4f}")

            # Refit on all data for the deployed artifact
            clf.fit(features, labels)
            return clf
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self, train_features_path):
        try:
            features = pd.read_csv(train_features_path)
            logging.info(f"Loaded features with shape {features.shape}")

            best_k = self.select_k(features)
            final_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
            labels = final_model.fit_predict(features)

            save_object(self.trainer_config.cluster_model_file_path, final_model)
            logging.info(f"Saved cluster model to {self.trainer_config.cluster_model_file_path}")

            profiles = self.profile_clusters(features, labels)
            os.makedirs(os.path.dirname(self.trainer_config.cluster_profiles_path), exist_ok=True)
            with open(self.trainer_config.cluster_profiles_path, "w") as f:
                json.dump(profiles, f, indent=2)
            logging.info(f"Saved cluster profiles to {self.trainer_config.cluster_profiles_path}")
            for cid, p in profiles.items():
                logging.info(f"  Cluster {cid} (n={p['size']}): {p['description']}")

            classifier = self.train_segment_classifier(features, labels)
            save_object(self.trainer_config.classifier_file_path, classifier)
            logging.info(f"Saved segment classifier to {self.trainer_config.classifier_file_path}")

            return self.trainer_config.classifier_file_path

        except Exception as e:
            raise CustomException(e, sys)