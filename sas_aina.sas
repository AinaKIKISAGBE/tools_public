

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$8oohkho8$$$$$$$$$$Mooo@$$$@oooW$$$$$$$*oM$@$$$*bbB$$$$$okhhoohkhk%$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$@$z       "n@$$$$$@$?   f$B$/   [$@$$$$$, <$$$J' ,UB$$$$$!         d$@$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$@$c  p%8ML  _$$$$$@$- .  M@#  . [$@$$$$$" ~$Zl .c@$$$$$$$B8%b  f%8%$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$@$c  k$$$$u  d$@$$@$? -/ 1${ f+ }$@$$$$$" ,~  'W$@@$$$$$$$$$o  x$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$@$c  b$$@$c  q$@$$@$? ~W  C  &i }$@$$$$$^   n! ^w$@$$$$$$$@$a  r$@$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$@$c  k$$Bb. i@$$$$@$? i$}   ($l }$@$$$$$" IB$%?  u$$$$$$$$$$a  r$@$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$@$v   ..   \%$$$$$@$- i$o   #$l [$@$$$$$^ <$B$$t  _M$$$$$$$$a  j$@$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$Wpdwmmp*$$$$$$$$$$opa$$hpa$$ap*$$$$$$$kpa$$$$$kdZM$$$$$$$$BdpM$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$@$@@@$$$$$$$$$$$$$@$$$$@$$$$@$$$$$$$$$@$$$$$$$@$$$$$$$$$$$$@$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$








oooooooooo.   ooo        ooooo oooo    oooo ooooooooooooo 
`888'   `Y8b  `88.       .888' `888   .8P'  8'   888   `8 
 888      888  888b     d'888   888  d8'         888      
 888      888  8 Y88. .P  888   88888[           888      
 888      888  8  `888'   888   888`88b.         888      
 888     d88'  8    Y     888   888  `88b.       888      
o888bood8P'   o8o        o888o o888o  o888o     o888o     







.sSSSSs.    .sSSSsSS SSsSSSSS .sSSS  SSSSS     .sSSSSSSSSs.   
SSSSSSSSSs. SSSSS  SSS  SSSSS SSSSS  SSSSS  .sSSSSSSSSSSSSSs. 
S SSS SSSSS S SSS   S   SSSSS S SSS SSSSS   SSSSS S SSS SSSSS 
S  SS SSSSS S  SS       SSSSS S  SS SSSSS   SSSSS S  SS SSSSS 
S..SS SSSSS S..SS       SSSSS S..SSsSSSSS   `:S:' S..SS `:S:' 
S:::S SSSSS S:::S       SSSSS S:::S SSSSS         S:::S       
S;;;S SSSSS S;;;S       SSSSS S;;;S  SSSSS        S;;;S       
S%%%S SSSS' S%%%S       SSSSS S%%%S  SSSSS        S%%%S       
SSSSSsS;:'  SSSSS       SSSSS SSSSS   SSSSS       SSSSS       










###########################################################
######### python generate asic art 1 from image png, jpg, ...  ########################
import sys
import numpy as np
from PIL import Image

# Contrast on a scale -10 -> 10
contrast = 10
density = ('$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|'
           '()1{}[]?-_+~<>i!lI;:,"^`\'.            ')
density = density[:-11+contrast]
n = len(density)


try:
    img_name = sys.argv[1]
    width = int(sys.argv[2])
except IndexError:
    # Default ASCII image width.
    width = 100
    img_name="C:/Users/user_distant/Downloads/DMKT.PNG"

# Read in the image, convert to greyscale.
img = Image.open(img_name)
img = img.convert('L')
# Resize the image as required.
orig_width, orig_height = img.size
r = orig_height / orig_width
# The ASCII character glyphs are taller than they are wide. Maintain the aspect
# ratio by reducing the image height.
height = int(width * r * 0.5)
img = img.resize((width, height), Image.ANTIALIAS)

# Now map the pixel brightness to the ASCII density glyphs.
arr = np.array(img)
for i in range(height):
    for j in range(width):
        p = arr[i,j]
        k = int(np.floor(p/256 * n))
        print(density[n-1-k], end='')
    print()

###########################################################
######### python generate asic art 2 from text ######################

def no_use():
    # -*- coding: utf-8 -*-
    """
    Created on Thu Sep 28 01:21:52 2023

    @author: user_distant
    """

    from art import *

    tprint("DMKT",font="rnd-large")


    tprint("DMKT","rnd-xlarge")



    text2art("DMKT",font="black") 

    import pyfiglet

    pyfiglet.figlet_format("DMKT",font='isometric1')


    pyfiglet.figlet_format("DMKT", font="3-d")

    pyfiglet.figlet_format("DMKT", font="alligator")


    DMKT DTM








@echo off
setlocal enabledelayedexpansion

set "input_file=mon_fichier.txt"
set "output_file=mon_fichier_modifie.txt"
set "search_string=chaine_a_trouver"
set "replacement_string=nouvelle_ligne"

if exist "%output_file%" del "%output_file%"

for /f "usebackq delims=" %%A in ("%input_file%") do (
    set "line=%%A"
    echo !line! | findstr /c:"%search_string%" >nul
    if !errorlevel! equ 0 (
        echo %replacement_string% >> "%output_file%"
    ) else (
        echo !line! >> "%output_file%"
    )
)

move /y "%output_file%" "%input_file%"
endlocal





xgboost

import optuna
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Charger un jeu de données exemple
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fonction objectif pour Optuna
def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0)
    }

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **param)
    score = cross_val_score(xgb_model, X_train, y_train, cv=3, scoring='accuracy').mean()
    
    return score

# Optimisation avec Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Afficher les meilleurs hyperparamètres
print(f"Meilleurs hyperparamètres : {study.best_params}")

# Évaluer le modèle avec les meilleurs hyperparamètres
best_params = study.best_params
best_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **best_params)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy}")






lighgbm

import optuna
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Charger un jeu de données exemple
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fonction objectif pour Optuna
def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 1.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 1.0)
    }

    lgb_model = lgb.LGBMClassifier(**param)
    score = cross_val_score(lgb_model, X_train, y_train, cv=3, scoring='accuracy').mean()
    
    return score

# Optimisation avec Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Afficher les meilleurs hyperparamètres
print(f"Meilleurs hyperparamètres : {study.best_params}")

# Évaluer le modèle avec les meilleurs hyperparamètres
best_params = study.best_params
best_model = lgb.LGBMClassifier(**best_params)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy}")



cv validation croise optuna xgboost 

import optuna
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score

# Générer un grand jeu de données déséquilibré
X, y = make_classification(n_samples=100000, n_features=20, n_informative=10, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fonction objectif pour Optuna avec validation croisée
def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1)  # Balance entre les classes
    }

    # Validation croisée stratifiée
    cv = StratifiedKFold(n_splits=5)
    cv_scores = []

    for train_idx, valid_idx in cv.split(X_train, y_train):
        X_cv_train, X_cv_valid = X_train[train_idx], X_train[valid_idx]
        y_cv_train, y_cv_valid = y_train[train_idx], y_train[valid_idx]
        
        dtrain_cv = xgb.DMatrix(X_cv_train, label=y_cv_train)
        dvalid_cv = xgb.DMatrix(X_cv_valid, label=y_cv_valid)
        
        bst = xgb.train(param, dtrain_cv, evals=[(dvalid_cv, 'eval')], early_stopping_rounds=50, verbose_eval=False)
        preds = bst.predict(dvalid_cv)
        pred_labels = [1 if p > 0.5 else 0 for p in preds]
        
        f1 = f1_score(y_cv_valid, pred_labels)
        cv_scores.append(f1)

    return sum(cv_scores) / len(cv_scores)

# Optimisation avec Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Afficher les meilleurs hyperparamètres
print(f"Meilleurs hyperparamètres : {study.best_params}")

# Entraînement final avec les meilleurs hyperparamètres
best_params = study.best_params
dtrain = xgb.DMatrix(X_train, label=y_train)
bst = xgb.train(best_params, dtrain, evals=[(xgb.DMatrix(X_test, label=y_test), 'eval')], early_stopping_rounds=50, verbose_eval=True)

# Évaluer le modèle
y_pred = bst.predict(xgb.DMatrix(X_test))
pred_labels = [1 if p > 0.5 else 0 for p in y_pred]
f1 = f1_score(y_test, pred_labels)
print(f"F1 Score : {f1}")




optuna cv validation croisé lightgbm 

import optuna
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score

# Générer un grand jeu de données déséquilibré
X, y = make_classification(n_samples=100000, n_features=20, n_informative=10, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fonction objectif pour Optuna avec validation croisée
def objective(trial):
    param = {
        'objective': 'binary',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 31, 300),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1)  # Balance entre les classes
    }

    # Validation croisée stratifiée
    cv = StratifiedKFold(n_splits=5)
    cv_scores = []

    for train_idx, valid_idx in cv.split(X_train, y_train):
        X_cv_train, X_cv_valid = X_train[train_idx], X_train[valid_idx]
        y_cv_train, y_cv_valid = y_train[train_idx], y_train[valid_idx]
        
        lgb_model = lgb.LGBMClassifier(**param)
        lgb_model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_valid, y_cv_valid)], early_stopping_rounds=50, verbose=False)
        
        y_cv_pred = lgb_model.predict(X_cv_valid)
        f1 = f1_score(y_cv_valid, y_cv_pred)
        cv_scores.append(f1)

    return sum(cv_scores) / len(cv_scores)

# Optimisation avec Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Afficher les meilleurs hyperparamètres
print(f"Meilleurs hyperparamètres : {study.best_params}")

# Entraînement final avec les meilleurs hyperparamètres
best_params = study.best_params
best_model = lgb.LGBMClassifier(**best_params)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Évaluer le modèle
f1 = f1_score(y_test, y_pred)
print(f"F1 Score : {f1}")







selection de seuil avec roc

from sklearn.metrics import roc_curve, auc

# Probabilités prédites par le modèle
y_pred_prob = bst.predict(dtest)

# Calculer la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calculer l'aire sous la courbe (AUC)
roc_auc = auc(fpr, tpr)

# Choisir le meilleur seuil
optimal_idx = np.argmax(tpr - fpr)  # Maximiser la différence
optimal_threshold = thresholds[optimal_idx]

print(f"Seuil optimal : {optimal_threshold}")




selection du seuil avec f1

from sklearn.metrics import f1_score

best_f1 = 0
best_threshold = 0.5

for threshold in np.arange(0.0, 1.0, 0.01):
    preds = (y_pred_prob >= threshold).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Meilleur seuil basé sur le F1 Score : {best_threshold}")






### chech memory

import psutil

# Récupérer des informations sur la mémoire
memory_info = psutil.virtual_memory()

# RAM totale
total_ram = memory_info.total / (1024 ** 3)  # Convertir en Go
print(f"RAM totale: {total_ram:.2f} Go")

# RAM disponible
available_ram = memory_info.available / (1024 ** 3)  # Convertir en Go
print(f"RAM disponible: {available_ram:.2f} Go")

# RAM utilisée
used_ram = memory_info.used / (1024 ** 3)  # Convertir en Go
print(f"RAM utilisée: {used_ram:.2f} Go")

# Pourcentage d'utilisation de la RAM
ram_usage_percentage = memory_info.percent
print(f"Pourcentage d'utilisation de la RAM: {ram_usage_percentage}%")




#########################################
import sys
import gc

# Fonction pour retourner la taille des objets Python en mémoire
def get_size(obj):
    """Renvoie la taille de l'objet en mémoire en octets."""
    return sys.getsizeof(obj)

# Récupérer tous les objets actuels en mémoire via le garbage collector
all_objects = gc.get_objects()

# Filtrer les objets qui consomment beaucoup de mémoire (par exemple > 1 Mo)
large_objects = [(type(obj), get_size(obj)) for obj in all_objects if get_size(obj) > 10**6]

# Trier par taille décroissante
large_objects_sorted = sorted(large_objects, key=lambda x: x[1], reverse=True)

# Afficher les objets avec leur type et leur taille en Mo
for obj_type, size in large_objects_sorted:
    print(f"Type: {obj_type}, Taille: {size / (1024 ** 2):.2f} Mo")





import numpy as np
from sklearn.model_selection import train_test_split

def train_test_split_with_single_class_handling(X, y, test_size=0.2, random_state=42):
    # Identifier les classes et leur nombre d'occurrences
    unique, counts = np.unique(y, return_counts=True)
    single_class = unique[counts == 1]
    
    # Indices des échantillons des classes avec un seul membre
    single_class_indices = np.where(np.isin(y, single_class))[0]
    
    # Indices des autres classes
    other_class_indices = np.where(~np.isin(y, single_class))[0]
    
    # Appliquer stratification sur les classes avec plus d'un membre
    X_train, X_test, y_train, y_test = train_test_split(
        X[other_class_indices], y[other_class_indices], 
        test_size=test_size, stratify=y[other_class_indices], 
        random_state=random_state
    )
    
    # Ajouter les échantillons uniques (un seul membre) à l'ensemble d'entraînement
    X_train = np.concatenate([X_train, X[single_class_indices]])
    y_train = np.concatenate([y_train, y[single_class_indices]])

    return X_train, X_test, y_train, y_test

# Exemple d'utilisation
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
y = np.array([0, 1, 1, 2, 2, 2, 3, 3, 4])  # Ici la classe 0 et 4 n'ont qu'un seul membre

X_train, X_test, y_train, y_test = train_test_split_with_single_class_handling(X, y)

print("X_train:", X_train)
print("y_train:", y_train)
print("X_test:", X_test)
print("y_test:", y_test)



import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_split_with_single_class_handling_df(X, y, test_size=0.2, random_state=42):
    # Identifier les classes et leur nombre d'occurrences
    class_counts = y.value_counts()
    single_class = class_counts[class_counts == 1].index
    
    # Indices des échantillons des classes avec un seul membre
    single_class_indices = y[y.isin(single_class)].index
    
    # Indices des autres classes
    other_class_indices = y[~y.isin(single_class)].index
    
    # Appliquer stratification sur les classes avec plus d'un membre
    X_train, X_test, y_train, y_test = train_test_split(
        X.loc[other_class_indices], y.loc[other_class_indices], 
        test_size=test_size, stratify=y.loc[other_class_indices], 
        random_state=random_state
    )
    
    # Ajouter les échantillons uniques (un seul membre) à l'ensemble d'entraînement
    X_train = pd.concat([X_train, X.loc[single_class_indices]])
    y_train = pd.concat([y_train, y.loc[single_class_indices]])

    return X_train, X_test, y_train, y_test

# Exemple d'utilisation avec des DataFrames
data = {'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
X = pd.DataFrame(data, index=[10, 11, 12, 13, 14, 15, 16, 17, 18])  # DataFrame avec des index aléatoires
y = pd.Series([0, 1, 1, 2, 2, 2, 3, 3, 4], index=[10, 11, 12, 13, 14, 15, 16, 17, 18])  # Série avec index aléatoires

X_train, X_test, y_train, y_test = train_test_split_with_single_class_handling_df(X, y)

print("X_train:\n", X_train)
print("y_train:\n", y_train)
print("X_test:\n", X_test)
print("y_test:\n", y_test)







################################## lift 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Fonction pour calculer le lift
def calculate_lift(y_true, y_pred, num_bins=10):
    # Créer un dataframe avec les vraies valeurs et les probabilités prédites
    data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    
    # Trier les données par les probabilités prédites (y_pred)
    data = data.sort_values('y_pred', ascending=False)
    
    # Diviser les données en segments égaux (déciles)
    data['bin'] = pd.qcut(data['y_pred'], q=num_bins, duplicates='drop')
    
    # Calculer le taux de réponse dans chaque segment
    lift_table = data.groupby('bin')['y_true'].agg(['sum', 'count'])
    lift_table['response_rate'] = lift_table['sum'] / lift_table['count']
    
    # Taux de réponse global
    overall_response_rate = data['y_true'].mean()
    
    # Calculer le lift (taux de réponse dans chaque segment divisé par le taux de réponse global)
    lift_table['lift'] = lift_table['response_rate'] / overall_response_rate
    
    return lift_table

# Fonction pour plot le lift chart
def plot_lift_chart(lift_table):
    plt.figure(figsize=(10, 6))
    
    # Tracer le lift dans chaque segment
    plt.plot(np.arange(1, len(lift_table) + 1), lift_table['lift'], marker='o', linestyle='-', color='b')
    
    # Tracer une ligne horizontale à y=1 (référence pour un modèle aléatoire)
    plt.axhline(y=1, color='r', linestyle='--')
    
    plt.title('Lift Chart')
    plt.xlabel('Deciles')
    plt.ylabel('Lift')
    plt.xticks(np.arange(1, len(lift_table) + 1))
    plt.grid(True)
    plt.show()

# Exemple d'utilisation
# y_true: vraies classes, y_pred: probabilités prédites
y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])  # Exemple de vraies classes
y_pred = np.array([0.05, 0.85, 0.15, 0.70, 0.90, 0.20, 0.95, 0.10, 0.60, 0.80])  # Exemple de probabilités prédites

# Calculer le lift
lift_table = calculate_lift(y_true, y_pred)

# Afficher le lift chart
plot_lift_chart(lift_table)





############################################## captation cumulé 

################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Exemple de données : scores du modèle et variable cible (0 ou 1)
data = {
    'score': [0.9, 0.85, 0.8,   0.6,   0.4, 0.3, 0.2, 0.1],
    'target': [1, 0, 1,         1,     0, 1,   0, 1]
}

# Créer un DataFrame à partir des données
df = pd.DataFrame(data)

# Trier les données par score décroissant
df = df.sort_values(by='score', ascending=False)

# Calculer la captation cumulée (cumul des 1 dans la colonne 'target')
df['cumulative_capture'] = df['target'].cumsum()

# Calculer le pourcentage de captation cumulative
df['cumulative_capture_pct'] = df['cumulative_capture'] / df['target'].sum() * 100

# Ajouter une colonne pour le pourcentage du nombre total d'observations
df['percent_observations'] = np.arange(1, len(df) + 1) / len(df) * 100

# Tracer le graphique de captation cumulée
plt.figure(figsize=(8, 6))
#plt.plot(df['percent_observations'], df['cumulative_capture_pct'], 
#         color='red', 
#         marker='o')
plt.bar(df['percent_observations'], df['cumulative_capture_pct'], 
        color='red', alpha=0.7)

# Ajouter des titres et labels
plt.title('Graphique de captation cumulée en fonction du score')
plt.xlabel('Pourcentage d\'observations')
plt.ylabel('Pourcentage de captation cumulée')

# Afficher le graphique
plt.grid(True)
plt.show()



####################################### hyperopt 

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# Charger et prétraiter les données
# Remplacez `data.csv` par le chemin de votre jeu de données
data = pd.read_csv("data.csv")
X = data.drop("target", axis=1)  # Remplacez "target" par le nom de la colonne cible
y = data["target"]

# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Sur-échantillonner l'ensemble d'entraînement pour traiter le déséquilibre
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Espace de recherche des hyperparamètres
space = {
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'gamma': hp.uniform('gamma', 0, 5),
    'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 10),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'reg_alpha': hp.loguniform('reg_alpha', -3, 3),
    'reg_lambda': hp.loguniform('reg_lambda', -3, 3)
}

# Fonction objectif pour Hyperopt
def objective(params):
    # Convertir les valeurs des hyperparamètres en entiers
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    params['min_child_weight'] = int(params['min_child_weight'])

    # Créer et entraîner le modèle
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        **params
    )

    model.fit(X_train_resampled, y_train_resampled, eval_set=[(X_val, y_val)], verbose=False)

    # Prédire sur l'ensemble de validation
    y_pred = model.predict_proba(X_val)[:, 1]
    
    # Calculer le score ROC AUC
    auc = roc_auc_score(y_val, y_pred)
    
    # Retourner le score pour Hyperopt
    return {'loss': -auc, 'status': STATUS_OK}

# Définir et lancer la recherche Hyperopt
trials = Trials()
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)

# Afficher les meilleurs hyperparamètres
print("Best hyperparameters:", best_params)








