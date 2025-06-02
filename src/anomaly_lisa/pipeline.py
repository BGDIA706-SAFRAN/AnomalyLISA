"""BGDIA706 Projet fil-rouge, SAFRAN.

Fichier pipeline.py : Défini le logger et le pipeline.

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
from typing import Final, Literal

# /* Modules externes */
# /* Module interne */


###############################################################################
# CONSTANTES :
EXIT_OK: Final[int] = 0

DEFAULT_SAVE_FOLDER: Final[str] = "results"
DEFAULT_LOG_FOLDER: Final[str] = "logs"
DEFAULT_LOG_FILENAME: Final[str] = "pipeline.log"
DEFAULT_MODEL_FOLDER: Final[str] = "checkpoints"


###############################################################################
# CLASSES PROJET :
class PipelineLogger:
    """Le logger du projet.

    ============
    Attributs :
        file                     le fichier de log si demandé ou stdout
        logger (logging.Logger)  le logger

    ============
    Méthodes :
        instance(msg, level,    **kwargs)  wrapper sur la classe logging.Logger
            pour logger des messages.
            => équivalent à instance.logger.info|error|...|(msg, **kwargs)
        setLevel(level, handler)    wrapper sur setLevel pour metter à jour le niveau
            de log des differents logger.

    ============
    Exemples :
        ```
        # CAS : fonctionnement manuel
        import Pipeline as TP
        a=TP.PipelineLogger()  # =>Seulement stdout
        a=TP.PipelineLogger(filepath='tt.txt')  # => sdtout + fichier
        a=TP.PipelineLogger(filepath='tt.txt', is_print=False)  # =>Seulement fichier
        a("test")
        a("erreur", level=logging.ERROR)
        a.setLevel(logging.INFO, "console")
        a.setLevel(logging.WARNING, "file")
        a("erreur", level=logging.INFO)  # Affiche en console et pas dans le fichier
        # CAS : automatique
        import Pipeline as TP
        args = {"nolog":False, "logfile":sys.stdout, "log_is_print":True, "log_folder":"log/timestamp/"}
        logger = TP.get_logger(args)
        ```
    """

    def __init__(self, filepath: str | None = None, is_print: bool = True,
                 logger_name: str = __name__):
        """Initialise la classe et le logger.

        :param (str) filepath:  fichier de log
        :param (bool) is_print: indique si les logs s'affiche aussi dans stdout
        """
        self.file = sys.stdout
        if filepath is not None:
            self.file = filepath

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "{asctime} - {name} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        if is_print:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            # console_handler.setLevel("DEBUG")
            logger.addHandler(console_handler)
        if isinstance(self.file, str):
            os.makedirs(os.path.dirname(self.file), exist_ok=True)
            file_handler = logging.FileHandler(self.file, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self.logger = logger

    def __call__(self, msg: str, level: int = logging.INFO, **kwargs):
        """Permet l'appel et wrapper sur la classe logging.Logger."""
        log_fct = {
            logging.DEBUG: self.logger.debug,
            logging.INFO: self.logger.info,
            logging.WARNING: self.logger.warning,
            logging.ERROR: self.logger.error,
            logging.CRITICAL: self.logger.critical,
        }
        log_fct.get(level, logging.INFO)(msg, **kwargs)

    def setLevel(self, level: int, handler: Literal["all", "console", "file", "root"] = "all"):
        """Permet de modifier le niveau de log des différentes instances.

        :param (int) level:     niveau de log (cf. `loggin.logger.setLevel`)
        :param (str) handler:   quel logger à modifier le niveua de log
            "console" => le logger en charge de la console si existe
            "file"    => le logger par fichier s'il existe
            "root"    => la valeur par défaut des logger si non spécifié
            "all"     => tout les logger
        """
        if handler in ("all", "root"):
            self.logger.setLevel(level)
        for x in self.logger.handlers:
            update = False
            is_file = isinstance(x, logging.FileHandler)
            is_console = isinstance(x, logging.StreamHandler) and not is_file

            update = (handler == "all") or \
                (handler == "console" and is_console) or \
                (handler == "file" and is_file)

            # Mise à jour du niveau de log
            if update:
                x.setLevel(level)


class Pipeline:
    """Le pipeline du projet.
    ...
    """

    name: str = "pipeline"

    def __init__(self, logger: PipelineLogger | None):
        """Initialise le pipeline."""
        self.logger = logger
        if logger is None:  # mode console
            self.logger = PipelineLogger()

    def test_sam_lisa(self):
        import ia_sam
        import ia_lisa
        from ia_sam import Agent_SAM
        from ia_lisa import Agent_LISA
        from PIL import Image, ImageDraw
        import numpy as np

        args = {}
        args["model_type"] = "vit_b"
        args["checkpoint"] = args.get("checkpoint", DEFAULT_MODEL_FOLDER)
        # args["device"] = args.get("device", ia_sam.DEVICE_GLOBAL)
        args["device"] = "cpu"

        agent_sam = Agent_SAM(self.logger, model_type=args["model_type"], checkpoint_path=args["checkpoint"], device=args["device"])

        args["input"] = "../data/MMAD/MVTec-AD/carpet/test/cut/000.png"  # "dogs.jpeg"
        args["bbox"] = "[65,250,631,940]"

        image_filename = args["input"]
        image_PIL = Image.open(image_filename)
        image_PIL = image_PIL.convert("RGB")
        image_rgb = np.array(image_PIL)
        local_arg = {
            "print": True,  # local print sans le logger
            "sam_img_in": image_rgb,
            "bbox": args["bbox"],
        }
        self.logger("Action : enlever le fond ...")
        agent_sam.run(local_arg, mode="background")
        agent_sam.save_results()

        args["input_prompt_str"] = "There is an defect ? explain in details"
        args["input_expert_str"] = ""
        args["version"] = args.get("version", ia_lisa.DEFAULT_LISA_MODEL)
        args["precision"] = args.get("precision", "bf16")
        args["image_size"] = args.get("image_size", 1024)
        args["load_in_8bit"] = args.get("load_in_8bit", False)
        args["load_in_4bit"] = args.get("load_in_4bit", False)
        args["model_max_length"] = args.get("model_max_length", ia_lisa.DEFAULT_LISA_MODEL_MAX_LENGTH)
        args["lora_r"] = args.get("lora_r", ia_lisa.DEFAULT_LISA_MODEL_LORA)
        args["vision_tower"] = args.get("vision_tower", ia_lisa.DEFAULT_LISA_MODEL_VISION_TOWER)
        args["use_mm_start_end"] = args.get("use_mm_start_end", ia_lisa.DEFAULT_LISA_MODEL_USE_MM_START_END)
        args["conv_type"] = args.get("conv_type", ia_lisa.DEFAULT_LISA_MODEL_CONV_TYPE)
        args["device"] = args.get("device", ia_lisa.DEVICE_GLOBAL)

        agent_lisa = Agent_LISA(args, logger=self.logger)

        local_arg["lisa_img_in"] = agent_sam.results["img_without_bg"]
        local_arg.update(args)
        agent_lisa.logger("Action : exécution LISA courante ...")
        agent_lisa.run(local_arg)
        agent.save_results()


###############################################################################
# CLASSES PROJET :
def get_logger(args: dict | None = None) -> PipelineLogger:
    """Créer et configure un logger.

    :param (dict) args:    les paramètres fournis
    :return (PipelineLogger):     le logger initialisé
    """
    ###
    # Gestion des arguments par défaut - compatible mode console ou mode Python
    ###
    if args is None:
        args = {}

    # nolog [--nolog] = (bool)
    args["nolog"] = args.get("nolog", False)
    # logfile [--logfile [FILENAME]] = défaut stdout, None si pas FILENAME
    args["logfile"] = args["logfile"] if "logfile" in args else sys.stdout

    filename = args["logfile"] if args["logfile"] is not None else DEFAULT_LOG_FILENAME
    foldername = os.path.join(DEFAULT_LOG_FOLDER, str(int(time.time())))
    # log_folder = (str) chemin du dossier de log
    args["log_folder"] = args.get("log_folder", foldername)
    # log_filepath = (str) None si pas de log en fichier
    if args["nolog"] or not isinstance(filename, str):
        args["log_filepath"] = None
    else:
        args["log_filepath"] = os.path.join(args["log_folder"], filename)
    # log_is_print = (bool) local pour afficher les log
    args["log_is_print"] = args.get("log_is_print", not args["nolog"])

    ###
    # Gestion du flux d'exécution
    ###
    # 0 - Création du logger
    # print(f"{args["task"]=}, {args["log_filepath"]=}, {args["log_is_print"]=}, {args["nolog"]=}, {args["logfile"]=}")
    logger = PipelineLogger(filepath=args["log_filepath"], is_print=args["log_is_print"])

    return logger


def run_process(args: dict | None = None) -> Pipeline:
    """Exécute le déroulé d'un pipeline.

    À modifier pour l'adapter à chaque agent IA

    :param (dict) args:    les paramètres fournis
    :return (PipelineLogger):     le logger initialisé
    """
    ###
    # Gestion des arguments par défaut - compatible mode console ou mode Python
    ###
    if args is None:
        args = {}

    ###
    # Gestion du flux d'exécution
    ###
    # 0 - Création du logger
    logger = get_logger(args)

    # 1 - Création du pipeline
    pipeline = Pipeline(logger)

    pipeline.test_sam_lisa()

    return pipeline


###############################################################################
# FONCTIONS MODE CONSOLE :
def parse_args() -> argparse.Namespace:
    """Gestion des arguments à modifier en fonction de l'agent IA.

    ====
    Arguments :
        logfile [--logfile [FILENAME]] = défaut stdout, None si pas FILENAME
        nolog [--nolog] = (bool)

    :param (str) args: les arguments données au programme
    :return (argparse.Namespace):   les arguments parsés
    """
    # 1 - Définition des listes de choix :
    # 2 - Création du parseur à arguments:
    parser = argparse.ArgumentParser(prog="Pipeline",
                                     description="Gère les différents enchaînement des agents.")
    # 3 - Définition des arguments :
    parser.add_argument("--logfile", nargs='?', type=argparse.FileType("w"),
                        default=sys.stdout, help="sortie du programme")
    parser.add_argument("--nolog", action='store_true',
                        help="désactive les log")
    parser.add_argument("--test_sam", action='store_true')
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
    pipeline = run_process(vars(args))
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
# TP.run({"task":"train"})
# TP.run({"nolog":False})
# TP.run({"nolog":True})
# TP.run({"logfile":sys.stderr})
# TP.run({"logfile":None})
# TP.run({"logfile":sys.stdout, "nolog":True})
# TP.run({"logfile":"tt.txt"})
# TP.run({"logfile":"tt.txt", "nolog":True})
# TP.run({"log_folder":"lll", "logfile":sys.stdout})
# TP.run({"log_folder":"lll", "logfile":"tt.txt"})
# TP.run({"log_is_print":False})
# TP.run({"log_is_print":False, "nolog":True})
# TP.run({"log_is_print":True})
# TP.run({"log_is_print":True, "nolog":True})
# TP.run({"logfile":"tt.txt", "log_filepath":"rr"})
