

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







