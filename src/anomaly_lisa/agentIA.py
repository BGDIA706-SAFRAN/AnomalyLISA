"""BGDIA706 Projet fil-rouge, SAFRAN.

Fichier agentIA.py : Défini un agent générique.

MEMO vérif code : python -m pylama -l all --pydocstyle-convention pep257 *.py
    # python -m pylama -l eradicate,mccabe,pycodestyle,pydocstyle,pyflakes,pylint,radon,vulture,isort --pydocstyle-convention pep257
    # python -m mypy
    # python -m autopep8 -a -p2 -r -v [-i | --diff]
    # python -m isort --diff
# MEMO console :
import importlib
importlib.reload(p)
"""
__author__ = ['Nicolas Allègre', 'Sarah Garcia', 'Luca Hachani', 'François-Xavier Morel']
__date__ = '17/05/2025'
__version__ = '0.2'

###############################################################################
# IMPORTS :
# /* Modules standards */
import argparse
import logging
import os
import sys
import time
from typing import Any, Final, Literal
from pprint import pprint

# /* Modules externes */
# /* Module interne */
import pipeline
from pipeline import PipelineLogger

###############################################################################
# CONSTANTES :
EXIT_OK: Final[int] = 0

DEFAULT_SAVE_MODEL_FILENAME: Final[str] = "agent_ia.pth"
DEFAULT_SAVE_RESULT_FILENAME: Final[str] = "agentIA_result"


###############################################################################
# CLASSES PROJET :
class AgentIA:
    """Classe interface modélisant un agent à dériver pour chaque agent IA.

    =============
    Attributs :
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
        import agent_IA as TP
        logger = TP.PipelineLogger()
        agent = TP.AgentIA(logger)
        args = ...
        agent.run(args)
        results = agent.results
        agent.save(mode="results", {"save_filepath":"save/agentX_results"})
    """

    name: str = "AgentIA"

    def __init__(self, logger: PipelineLogger | None = None):
        """Initialise cette classe générique."""
        self.model = None
        self.results: dict[str, Any] = {}
        self.logger = logger
        if logger is None:  # mode console
            self.logger = PipelineLogger(logger_name=__name__)

    def run(self, args: dict | None = None):
        """Exécute la tâche de l'agent IA.

        :param (dict) args: les arguments de la fonction
            args["print"] affiche des infos (if args.get("print", False): #print)
        """
        results: dict[str, Any] = {}
        if args is None:
            args = {}
        is_print = args.get("print", False)
        if is_print:
            print("Exécution de ", self.name)

        self.results = results

    def train(self):
        """Entraînement le modèle de l'agent IA."""
        return

    def create_prompt(self, args: dict | None = None):
        """Crée le prompt expert associé à la tâche.

        :param (dict) args: les arguments de la fonction
        """
        prompt = ""
        if len(self.results) == 0:
            self.logger(f"Pas de résultat pour créer le prompt en {mode}", level="error")
            return

        # Faire quelque pour créer le prompt.
        self.results["prompt"] = prompt
        return

    def save(self, mode: Literal["all", "model", "results"] = "all",
             args: dict | None = None):
        """Sauvegarde le modèle entraîné ou les résultats.

        :param (str) mode:  spécifie quoi enregistrer ["all", "model", "results"]
        :param (dict) args: les arguments d'enregistrement
            args["save_filepath"]
            args["save_is_print"]
        """
        if mode in ("all", "model"):
            self.save_model(args)
        if mode in ("all", "results"):
            self.save_results(args)
        return

    def save_model(self, args: dict | None = None):
        """Sauvegarde le modèle.

        :param (dict) args: les arguments pour la sauvegarde du modèle
            args["model_save_filename"]
            args["model_save_folder"]
        """
        if args is None:
            args = {}
        filename = args.get("model_save_filename", DEFAULT_SAVE_MODEL_FILENAME)
        foldername = args.get("model_save_folder", pipeline.DEFAULT_SAVE_FOLDER)
        return

    def save_results(self, args: dict | None = None):
        """Sauvegarde les résultats.

        :param (dict) args: les arguments pour la sauvegarde du modèle
            args["results_save_filename"]
            args["results_save_folder"]
        """
        if args is None:
            args = {}
        filename = args.get("results_save_filename", DEFAULT_SAVE_RESULT_FILENAME)
        foldername = args.get("results_save_folder", pipeline.DEFAULT_SAVE_FOLDER)
        # foldername = os.path.join(foldername, "agentX")  # Si beaucoup de fichier résultats
        os.makedirs(foldername, exist_ok=True)
        prompt_filename = f"{filename}_prompt.txt"  # exemple de construction du nom de fichier
        return


###############################################################################
# CLASSES PROJET :
def run_process(args: dict | None = None, logger: PipelineLogger | None = None) -> AgentIA:
    """Exécute le déroulé d'une tâche.

    À modifier pour l'adapter à chaque agent IA

    :param (dict) args:    les paramètres fournis
    :return (AgentIA):     l'agent utilisé avec ses résultats et modèle
    """
    ###
    # Gestion des arguments par défaut - compatible mode console ou mode Python
    ###
    if args is None:
        args = {}

    # task [--task TASK] = (str) "run", "train", ...
    args["task"] = args.get("task", "run")
    # nolog [--nolog] = (bool)
    args["nolog"] = args.get("nolog", False)
    # logfile [--logfile [FILENAME]] = défaut stdout, None si pas FILENAME
    args["logfile"] = args["logfile"] if "logfile" in args else sys.stdout
    # savefile [--savefile [FILENAME]] = défaut stdout, None si pas FILENAME
    args["savefile"] = args["savefile"] if "savefile" in args else sys.stdout
    # checkpoint [--checkpoint FOLDER] = (str)
    args["checkpoint"] = args.get("checkpoint", pipeline.DEFAULT_MODEL_FOLDER)
    # device [--device DEVICE] = (str) 'auto, cpu, cuda, torch_directml'
    # args["device"] = args.get("device", utils.torch_pick_device())
    args["device"] = args.get("device", "cpu")
    # output [--output [FOLDER]] = (str) défaut 'results'
    args["output"] = args.get("output", pipeline.DEFAULT_SAVE_FOLDER)

    filename = args["logfile"] if args["logfile"] is not None else AgentIA.name
    foldername = os.path.join(pipeline.DEFAULT_LOG_FOLDER, str(int(time.time())))
    # log_filepath = (str) None si pas de log en fichier
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
        # Seul train est obligé de sauvegarder dans un fichier les autres au choix
        if args["task"] == "train" and filename is None:
            filename = DEFAULT_SAVE_MODEL_FILENAME
        if filename is not None:
            args["save_filepath"] = os.path.join(foldername, filename)
        else:
            args["save_is_print"] = True

    ###
    # Gestion du flux d'exécution
    ###
    # pprint(args)  # pour test
    # 0 - Création du logger
    if logger is None:
        # logger = pipeline.get_logger(args)
        logger = PipelineLogger(filepath=args["log_filepath"], is_print=args["log_is_print"], logger_name=AgentIA.name)

    # 1- Création et configuration agent
    agent = AgentIA(logger)

    # 2- Suivant la tâche exécution de celle-ci
    local_arg = {}
    if args["task"] == "run":
        # CAS 1 : Si on est sûr aucun conflit
        # agent.run(args)

        # CAS 2 : args spécifiques à la tâche
        local_arg = {"print": True}  # local print sans le logger
        agent.logger("Action : exécution xxx ...")
        agent.run(local_arg)
    elif args["task"] == "train":
        agent.logger("Action : entraînement ...")
        agent.train()

    # 3- Sauvegarde
    modes: dict[str, str] = {"run": "results", "train": "model"}
    mode: str = modes.get(args["task"], "all")

    local_arg["save_filepath"] = args["save_filepath"]
    local_arg["save_is_print"] = args["save_is_print"]
    if is_saving:
        agent.logger("Action : saving ...")
        agent.save(mode, local_arg)

    return agent


###############################################################################
# FONCTIONS MODE CONSOLE :
def parse_args() -> argparse.Namespace:
    """Gestion des arguments à modifier en fonction de l'agent IA.

    ====
    Arguments :
        task [--task TASK] = (str) "run", "train", ...
        logfile [--logfile [FILENAME]] = défaut stdout, None si pas FILENAME
        savefile [--savefile [FILENAME]] = défaut stdout, None si pas FILENAME
        nolog [--nolog] = (bool)

    :param (str) args: les arguments données au programme
    :return (argparse.Namespace):   les arguments parsés
    """
    # 1 - Définition des listes de choix :
    list_task_agentIA = ["run", "train"]

    # 2 - Création du parseur à arguments:
    parser = argparse.ArgumentParser(prog="AgentIA X",
                                     description="Ce que l'agent IA fait.")

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
    parser.add_argument("--device", type=str, default="cpu",
                        help="[défaut=cpu] device où charger [auto, cpu, cuda, torch_directml]")
    parser.add_argument("--output", type=str, nargs='?', default=pipeline.DEFAULT_SAVE_FOLDER,
                        help="[défaut=results] chemin du dossier de sortie")
    parser.add_argument("--checkpoint", type=str, default=pipeline.DEFAULT_MODEL_FOLDER,
                        help="[défaut=checkpoints] dossier où sont les poids du modèle")

    # 3.2 args spécifique LISA
    #

    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    """Exécute le flux d'exécution des tâches pour l'agent.

    :param args:    les paramètres fournis en ligne de commande
    :return (int):  le code retour du programme
    """
    exit_value: int = EXIT_OK
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

####
# TEST pour mémo
# TP.run()
# TP.run({"savefile":sys.stderr})
# TP.run({"savefile":None})
# TP.run({"savefile":"tt.txt"})
# TP.run({"savefile":"tt.txt", "save_filepath":"tttttt"})
# TP.run({"save_folder":"lll", "savefile":sys.stdout})
# TP.run({"save_folder":"lll", "savefile":"tt.txt"})
# TP.run({"task":"train"})
# TP.run({"task":"train", "savefile":None})
# TP.run({"task":"train", "savefile":None, "save_is_print":True})
# TP.run({"task":"train", "savefile":"tt.txt"})
# TP.run({"save_is_print":False, "savefile":None})
# TP.run({"save_is_print":False, "savefile":"ttg.txt"})
