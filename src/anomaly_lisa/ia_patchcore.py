"""BGDIA706 Projet fil-rouge, SAFRAN.

Fichier ia_patchcore.py : Défini l'agent Patchcore

Il fonctionne en ligne de commande ou en appel :
    - ia_sam.run_process(**args**, [logger])
    - python **args**

Où **args** sont a minima :
    python ia_patchcore.py -h
    ...

Et en mode exécution Python des args optionnels supplémentaires sont :
 - ...

===========
Exemples :
...

"""
__author__ = ['Nicolas Allègre', 'Sarah Garcia', 'Luca Hachani', 'François-Xavier Morel']
__date__ = '26/05/2025'
__version__ = '0.1'

###############################################################################
# IMPORTS :
# /* Modules standards */
import argparse
import ast
import importlib
import logging
import os
import sys
import time
from pprint import pprint
from typing import Any, Final, Literal

# /* Modules externes */
import numpy as np
import torch
from PIL import Image, ImageDraw
from anomalib.callbacks import LoadModelCallback
from anomalib.data import MVTecAD, Visa
from anomalib.engine import Engine
from anomalib.models import Patchcore

# /* Module interne */
import pipeline
from agentIA import AgentIA
from pipeline import PipelineLogger

###############################################################################
# CONSTANTES :
EXIT_OK: Final[int] = 0

DEFAULT_SAVE_MODEL_FILENAME: Final[str] = "patchcore_local.pth"
DEFAULT_SAVE_RESULT_FILENAME: Final[str] = "PATCHCORE"

DEVICE_GLOBAL: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# CLASSES PROJET :
class Agent_Patchcore(AgentIA):
    """Classe de l'agent Patchcore

    =============
    Attributs (mère) :
        name      (class) le nom de l'agent
        model     l'IA de l'agent IA
        results   les résultats de l'exécution
        logger    le logger pour les messages

    =============
    Méthodes
        run(args: dict())   Exécute la tâche de l'agent IA.
        train()             Entraînement le modèle de l'agent IA.
        save(mode, args)    Sauvegarde du modèle entraîné ou les résultats.

    =============
    Exemples :
        ...
    """

    name: str = "PATCHCORE"

    def __init__(self, args: dict, logger: PipelineLogger | None = None):
        """Initialise cette classe."""
        super().__init__(logger)
        self.models: dict = self.load(args)

    def load(self, args: dict) -> dict:
        """Charge en mémoire le modèle Patchcore.

        :param (dict) args: les arguments nécessaire
            args["patchcore_args"]
            args["patchcore_engine_args"]
            args["results_save_folder"]
            args["ckpt_path"]
        :return (dict): les éléments du modèle
            results["model"] = model Patchcore anomalib
            results["engine"] = engine anomalib
            results["ckpt_path"] = chemin vers les checkpoints entraînés de Patchcore
        """
        device = args.get("device", DEVICE_GLOBAL)
        results = {}
        if args is None:
            args = {}

        foldername = args.get("results_save_folder", pipeline.DEFAULT_SAVE_FOLDER)
        args["ckpt_path"] = args.get("ckpt_path", pipeline.DEFAULT_MODEL_FOLDER)
        # Valeurs par défauts de Patchcore :
        args["patchcore_args"] = args.get("patchcore_args", {})
        args_model = args["patchcore_args"]
        args_model["backbone"] = args_model.get("backbone", 'wide_resnet50_2')
        args_model["layers"] = args_model.get("layers", ('layer2', 'layer3'))
        args_model["pre_trained"] = args_model.get("pre_trained", True)
        args_model["coreset_sampling_ratio"] = args_model.get("coreset_sampling_ratio", 0.1)
        args_model["num_neighbors"] = args_model.get("num_neighbors", 9)
        args["patchcore_args"] = args_model
        # Valeurs par défauts de l'Engine
        args["patchcore_engine_args"] = args.get("patchcore_engine_args", {})
        args_engine = args["patchcore_engine_args"]
        args_engine["default_root_dir"] = args_engine.get("default_root_dir", foldername)  # anomalib => 'results'
        args["patchcore_engine_args"] = args_engine

        # 2. Initialize the model and load weights
        model = Patchcore(**args_model)
        engine = Engine(**args_engine)

        results["model"] = model
        results["engine"] = engine
        results["ckpt_path"] = args["ckpt_path"]
        return results

    def train(self, args: dict, datamodule_class=Visa, category: str = "capsules",
              save: bool = True, filename: str = "") -> Patchcore:
        """Entraine un patchcore sur une catégorie d'un dataset et sauvegarde.

        :param datamodule_class (anomalib.data.datamodules): Dataset utilisé (Visa, MVTecAD)
        :param category (str):  catégorie du dataset à entraîner
        :param save (bool): spécifie s'il faut sauvegarder
        :param filename (str):  chemin et nom du fichier à sauvegarder le modèle
        :return (anomalib.models.patchcore):    le patchcore entraîné
        """
        datamodule = datamodule_class(category=category)
        model = self.models["model"]
        engine = self.models["engine"]
        engine.fit(model=model, datamodule=datamodule)
        if save:
            # sinon engine.export(model=model, export_type=ExportType.TORCH, export_root=foldername, model_file_name=f"model_category") ??
            torch.save(model.state_dict(), filename)

        return model

    # Rappel classe mère AgentIA :
    # def run(self, args: dict)
    # def train(self)
    # def save(self, mode: Literal["all", "model", "results"] = "all", args: dict | None = None)
    # def save_model(self, args: dict | None = None)
    # def save_results(self, args: dict | None = None)
    # anomalib.models.image.patchcore.anomaly_map.AnomalyMapGenerator


###############################################################################
# FONCTION PROJET :
def run_process(args: dict | None = None, logger: PipelineLogger | None = None) -> Agent_Patchcore:
    """Exécute le déroulé d'une tâche.

    Déroule le fonctionnement de l'agent Patchcore.
        run :           exécute Patchcore ...

    :param (dict) args:    les paramètres fournis
    :return (Agent_Patchcore):     l'agent utilisé avec ses résultats et modèle
    """
    ###
    # Gestion des arguments par défaut - compatible mode console ou mode Python
    ###
    if args is None:
        args = {}

    # 3.1 args génériques
    # task [--task TASK] = (str) "run", "train", ...
    args["task"] = args.get("task", "run")
    # nolog [--nolog] = (bool)
    args["nolog"] = args.get("nolog", False)
    # logfile [--logfile [FILENAME]] = défaut stdout, None si pas FILENAME
    args["logfile"] = args.get("logfile", sys.stdout)
    # savefile [--savefile [FILENAME]] = défaut stdout, None si pas FILENAME
    args["savefile"] = args.get("savefile", sys.stdout)
    # log_filepath = (str) None si pas de log en fichier
    filename = args["logfile"] if args["logfile"] is not None else f"{Agent_Patchcore.name}.log"
    foldername = os.path.join(pipeline.DEFAULT_LOG_FOLDER, str(int(time.time())))
    if args["nolog"] or not isinstance(filename, str):
        args["log_filepath"] = None
    else:
        folder = args.get("log_folder", foldername)
        args["log_filepath"] = os.path.join(folder, filename)
    # log_is_print = (bool) local pour afficher les log
    args["log_is_print"] = args.get("log_is_print", not args["nolog"])

    is_saving = False
    args["save_filepath"] = None
    args["save_is_print"] = args.get("save_is_print", False)
    if args["savefile"] is None or isinstance(args["savefile"], str):  # SAVE :
        is_saving = True
        filename = args["savefile"]
        foldername = args.get("save_folder", pipeline.DEFAULT_SAVE_FOLDER)
        if filename is None:
            if args["task"] == "train":
                filename = DEFAULT_SAVE_MODEL_FILENAME
            else:
                filename = DEFAULT_SAVE_RESULT_FILENAME
        if filename is not None:
            args["save_filepath"] = os.path.join(foldername, filename)
        else:
            args["save_is_print"] = True

    # 3.2 args spécifique Patchcore
    # ...

    ###
    # Gestion du flux d'exécution
    ###
    pprint(args)  # pour test
    # 0 - Création du logger
    if logger is None:
        # logger = pipeline.get_logger(args)
        logger = PipelineLogger(filepath=args["log_filepath"], is_print=args["log_is_print"], logger_name=Agent_SAM.name)

    # 1- Création et configuration agent
    # agent = Agent_Patchcore(...)
    agent = Agent_Patchcore

    # 2- Suivant la tâche exécution de celle-ci
    # ...

    # 3-Sauvegarde
    # ...

    return agent


###############################################################################
# FONCTIONS MODE CONSOLE :
def parse_args() -> argparse.Namespace:
    """Gestion des arguments de l'agent Patchcore.

    ====
    Arguments :
        task [--task TASK] = (str) "run", "train", "background"
        logfile [--logfile [FILENAME]] = défaut stdout, None si pas FILENAME
        savefile [--savefile [FILENAME]] = défaut stdout, None si pas FILENAME
        nolog [--nolog] = (bool) désactive les logs
        checkpoint [--checkpoint FOLDER] = (str)
        device [--device DEVICE] = (str) 'auto, cpu, cuda, torch_directml'
        ...

    :param (str) args: les arguments données au programme
    :return (argparse.Namespace):   les arguments parsés
    """
    # 1 - Définition des listes de choix :
    list_task_agentIA = ["run", "train", "background"]

    # 2 - Création du parseur à arguments:
    parser = argparse.ArgumentParser(prog="AgentIA Patchcore",
                                     description="command line Patchcore")

    # 3 - Définition des arguments :
    # 3.1 args génériques
    parser.add_argument("--logfile", nargs='?', type=argparse.FileType("w"),
                        default=sys.stdout, help="sortie du programme")
    parser.add_argument("--savefile", nargs='?', type=argparse.FileType("w"),
                        default=sys.stdout,
                        help="""\
                            spécifie s'il faut sauvegarder.
                            Si aucun fichier, alors stdout sauf pour 'train'
                            """)
    parser.add_argument("--task", type=str, choices=list_task_agentIA, default="run",
                        help="[défaut=run] tâche à accomplir par l'agent")
    parser.add_argument("--nolog", action='store_true',
                        help="désactive les log")
    parser.add_argument("--checkpoint", type=str, default=pipeline.DEFAULT_MODEL_FOLDER,
                        help="[défaut=checkpoints] dossier où sont les poids")
    parser.add_argument("--device", type=str, default="cpu",
                        help="[défaut=auto] device où charger le modèle [auto, cpu, cuda, torch_directml] (gérer par anomalib)")

    # 3.2 args spécifique Patchcore
    # ...

    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    """Exécute le flux d'exécution des tâches pour l'agent.

    :param args:    les paramètres fournis en ligne de commande
    :return (int):  le code retour du programme
    """
    exit_value = EXIT_OK
    # Gestion particulière des args externes si besoin
    print(args)

    # Exécution
    start_time = time.time()
    agent = run_process(vars(args))
    print(f'\n--- {time.time() - start_time} seconds ---')

    return exit_value


###############################################################################
if __name__ == "__main__":
    print(sys.argv)
    sys.exit(main(parse_args()))
# end if
