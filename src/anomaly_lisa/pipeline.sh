#! /bin/sh

###############################################################################
# CONSTANTES :
################
# Les différents types d'environnement Python
GLOBAL="myenv_anomalib"
LISA="myenv_lisa"

# Les chemins globaux
miniconda="~/miniconda3/bin/activate"
################

###############################################################################
# FONCTIONS :
function read_config_file() {
    # Lecture d'un fichier d'entrée de config afin de savoir quel pipeline à exécuter.
    # (sinon à récupérer par les arguments en ligne de commande)
    # $1 = nom du fichier config
    # return => $? = les informations d'exécution (quel agent, quel enchaînement, quel fichier IN, etc.)
}

function read_config_agent() {
    # Lecture du fichier conf d'un agent
    # $1 = nom du fichier config
    # return => $? = les informations nécessaire à lancer l'agent (comme son env Python)
}

###############################################################################
# 0 - Gestion des arguments
config_path = $1

# 1 - Activation de l'environnement Python
. ~/miniconda3/bin/activate

# 2 - Récupértion des tâches à accomplir et du pipeline
config_pipeline=read_config_file $config_path

# 3 - Récupération des différents conf des agents à exécuter
# for chaque agent demandé charger son fichier de conf

# 4 - Décision d'action à faire


# Exemple concept fonctionnement partie python

################
## ENV ANOMALIB
conda activate $GLOBAL

python --version
python -m pip list |grep -E "numpy|torch"
python pipeline.py -h
python ia_sam.py --input=dog.jpeg --model-type=vit_b --bbox=[65,250,631,940] --task background --savefile --logfile

conda deactivate
################
################
## ENV LISA
conda activate $LISA
cd  experiments/LISA/
python --version
python -m pip list |grep -E "numpy|torch"
ls -l chat.py

conda deactivate
################

# 99 - Dé-activation de l'environnement Python
conda deactivate