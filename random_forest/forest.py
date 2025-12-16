import numpy as np
import sys
import os

# ajoute le répertoire parent au chemin pour importer DecisionTree
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from decision_tree import DecisionTree
except ImportError:
    # affiche une erreur si le module DecisionTree ne peut pas être trouvé
    raise ImportError("impossible de trouver le module 'decision_tree'. vérifie la structure de tes dossiers.")

class RandomForest:
    """
    implémentation de la forêt aléatoire basée sur breiman (2001).
    supporte à la fois la classification et la régression.
    """

    def __init__(
        self,
        n_estimators=100,           # nombre d'arbres dans la forêt
        max_depth=None,             # profondeur maximale de chaque arbre
        min_samples_split=2,        # nombre minimum d'échantillons pour diviser un nœud
        min_samples_leaf=1,         # nombre minimum d'échantillons dans une feuille
        max_features='sqrt',        # nombre de caractéristiques à considérer par split
        bootstrap=True,             # utilise l'échantillonnage bootstrap
        oob_score=False,            # calcule l'erreur hors sac (out-of-bag)
        random_state=None           # graine aléatoire pour la reproductibilité
    ):
        # stocke les hyperparamètres
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        
        # initialise les structures de données
        self.trees = []                             # liste pour stocker les arbres entraînés
        self.feature_importances_ = None            # importance de chaque caractéristique
        self.is_classifier = True                   # détecté automatiquement pendant l'ajustement

    def fit(self, X, y):
        """
        entraîne la forêt en utilisant le bagging (échantillonnage bootstrap).
        
        paramètres:
            X: array de caractéristiques (n_samples, n_features)
            y: array cible (classification ou régression)
        """
        n_samples, n_features = X.shape
        self.trees = []
        
        # détecte automatiquement le type de tâche (classification ou régression)
        # si y contient des floats, c'est de la régression; sinon, classification
        if np.issubdtype(y.dtype, np.floating):
            self.is_classifier = False
            criterion = 'mse'  # utilise erreur quadratique moyenne pour la régression
        else:
            self.is_classifier = True
            criterion = 'gini'  # utilise l'indice gini pour la classification

        # crée un générateur de nombres aléatoires avec la graine spécifiée
        rng = np.random.default_rng(self.random_state)
        
        # initialise les structures pour le calcul de l'erreur hors sac si demandé
        if self.oob_score and self.bootstrap:
            if self.is_classifier:
                # compte les votes par classe pour chaque échantillon
                unique_classes = np.unique(y)
                oob_preds = np.zeros((n_samples, len(unique_classes)))
            else:
                # accumule les prédictions de régression
                oob_preds = np.zeros(n_samples)
                
            oob_counts = np.zeros(n_samples)  # compte combien de fois chaque échantillon est hors sac

        # initialise l'importance des caractéristiques à zéro
        self.feature_importances_ = np.zeros(n_features)

        # boucle principale: crée et entraîne chaque arbre
        for i in range(self.n_estimators):
            
            # étape 1: échantillonnage bootstrap (avec remise)
            if self.bootstrap:
                # sélectionne n_samples indices avec remise
                idxs = rng.choice(n_samples, size=n_samples, replace=True)
                X_sample = X[idxs]
                y_sample = y[idxs]
            else:
                # utilise l'ensemble du dataset sans bootstrap
                X_sample, y_sample = X, y

            # étape 2: crée et entraîne un arbre de décision
            # génère une graine aléatoire unique pour reproduire les résultats
            tree_seed = rng.integers(0, 10**9)
            
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,  # gère la sélection aléatoire de caractéristiques
                criterion=criterion,
                random_state=tree_seed
            )
            
            # entraîne l'arbre sur les données échantillonnées
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
            # étape 3: accumule les importances de caractéristiques de chaque arbre
            if hasattr(tree, 'feature_importances_'):
                self.feature_importances_ += tree.feature_importances_

            # étape 4: gère les prédictions hors sac pour l'évaluation du modèle
            if self.oob_score and self.bootstrap:
                # trouve les indices qui n'ont pas été sélectionnés dans le bootstrap
                mask = np.ones(n_samples, dtype=bool)
                mask[idxs] = False
                oob_idxs = np.where(mask)[0]
                
                if len(oob_idxs) > 0:
                    # prédictions sur les données hors sac
                    preds = tree.predict(X[oob_idxs])
                    
                    if self.is_classifier:
                        # enregistre les votes pour chaque classe
                        for idx, p in zip(oob_idxs, preds):
                            oob_preds[idx, int(p)] += 1
                    else:
                        # accumule les valeurs prédites pour la régression
                        oob_preds[oob_idxs] += preds
                        oob_counts[oob_idxs] += 1

        # normalise les importances en les divisant par le nombre d'arbres
        self.feature_importances_ /= self.n_estimators
        
        # calcule et enregistre le score final hors sac
        if self.oob_score and self.bootstrap:
            self._finalize_oob_score(y, oob_preds, oob_counts)
            
        return self

    def predict(self, X):
        """
        fait des prédictions en utilisant:
        - le vote majoritaire pour la classification
        - la moyenne pour la régression
        
        paramètres:
            X: array de caractéristiques (n_samples, n_features)
        
        retourne:
            array de prédictions
        """
        if not self.trees:
            raise Exception("le modèle n'a pas été entraîné encore.")

        # obtient les prédictions de tous les arbres
        # la forme devient (n_samples, n_estimators) après transposition
        all_preds = np.array([tree.predict(X) for tree in self.trees]).T
        
        if self.is_classifier:
            # vote majoritaire: trouve la classe la plus fréquente pour chaque échantillon
            final_preds = []
            for row in all_preds:
                # bincount compte les occurrences de chaque classe et argmax trouve la plus fréquente
                final_preds.append(np.bincount(row.astype(int)).argmax())
            return np.array(final_preds)
        else:
            # pour la régression: calcule la moyenne des prédictions
            return np.mean(all_preds, axis=1)

    def predict_proba(self, X):
        """
        obtient les probabilités de classe (moyenne des votes).
        fonctionne uniquement pour la classification.
        
        paramètres:
            X: array de caractéristiques (n_samples, n_features)
        
        retourne:
            array de probabilités de forme (n_samples, n_classes)
        """
        if not self.is_classifier:
            raise ValueError("predict_proba est réservée à la classification.")
            
        # obtient les prédictions de tous les arbres
        all_preds = np.array([tree.predict(X) for tree in self.trees]).T
        n_samples = X.shape[0]
        
        # détermine le nombre de classes à partir des prédictions
        n_classes = int(np.max(all_preds) + 1)
        probas = np.zeros((n_samples, n_classes))
        
        # pour chaque échantillon, compte les votes par classe
        for i in range(n_samples):
            counts = np.bincount(all_preds[i].astype(int), minlength=n_classes)
            probas[i] = counts / self.n_estimators
            
        return probas

    def _finalize_oob_score(self, y, oob_preds, oob_counts):
        """
        méthode auxiliaire pour calculer le score final hors sac.
        pour la classification: calcule la précision (accuracy)
        pour la régression: calcule le coefficient de détermination (r²)
        
        paramètres:
            y: valeurs cibles réelles
            oob_preds: prédictions accumulées pour les données hors sac
            oob_counts: nombre de fois que chaque échantillon était hors sac
        """
        if self.is_classifier:
            # sélectionne la classe avec le maximum de votes
            final_oob_preds = np.argmax(oob_preds, axis=1)
            
            # ne considère que les échantillons qui ont été hors sac au moins une fois
            valid = np.sum(oob_preds, axis=1) > 0
            if np.any(valid):
                # calcule la précision sur les prédictions valides
                self.oob_score_ = np.mean(y[valid] == final_oob_preds[valid])
            else:
                self.oob_score_ = 0.0
        else:
            # évite la division par zéro
            oob_counts[oob_counts == 0] = 1 
            # calcule la moyenne des prédictions en les divisant par le nombre de votes
            final_oob_preds = oob_preds / oob_counts
            
            # calcule le coefficient de détermination (r²)
            # u: somme des carrés des erreurs
            u = ((y - final_oob_preds) ** 2).sum()
            # v: somme totale des carrés (variance totale)
            v = ((y - y.mean()) ** 2).sum()
            
            if v == 0:
                self.oob_score_ = 0.0
            else:
                # r² = 1 - (u/v)
                self.oob_score_ = 1 - (u / v)