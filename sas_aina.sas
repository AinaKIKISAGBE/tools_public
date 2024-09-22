

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










