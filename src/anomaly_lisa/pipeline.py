"""BGDIA706 Projet fil-rouge, SAFRAN.

Fichier pipeline.py : Défini le logger et le pipeline.

Il fonctionne en ligne de commande ou en appel :
    - ia_sam.run_process(**args**, [logger])
    - python **args**

Où **args** sont a minima :
    python pipeline.py -h
['pipeline.py', '-h']
usage: Pipeline [-h] [--logfile [LOGFILE]] [--savefile [SAVEFILE]] [--task {run}] [--nolog] [--checkpoint CHECKPOINT]
                [--device DEVICE] [--test_sam] [--agents AGENTS] [--agents_args AGENTS_ARGS [AGENTS_ARGS ...]]
                [--agents_map AGENTS_MAP [AGENTS_MAP ...]]

Gère les différents enchaînement des agents.

options:
  -h, --help            show this help message and exit
  --logfile [LOGFILE]   sortie du programme
  --savefile [SAVEFILE]
                                                    spécifie s'il faut sauvegarder.
                                                    Si aucun fichier, alors stdout sauf pour 'train'

  --task {run}          [défaut=run] tâche à accomplir par l'agent
  --nolog               désactive les log
  --checkpoint CHECKPOINT
                        [défaut=checkpoints] dossier où sont les poids des modèles.
  --device DEVICE       [défaut=cpu] device où charger le modèle [auto, cpu, cuda, torch_directml]
  --test_sam
  --agents AGENTS       Liste ordonnée d'agent à exécuter. (agent1,agent2)
  --agents_args AGENTS_ARGS [AGENTS_ARGS ...]
                        Liste ordonnée des args pour chaque agent en forme dict (comme en ligne de commande).
                            "AGENT_NAME=<args>;AGENT_NAME=<args>;..."
                            --> "<args> : arg_name:value,arg_name_whitout,..."

  --agents_map AGENTS_MAP [AGENTS_MAP ...]
                        Liste ordonnée d'agent à exécuter. (agent1,agent2)

Exemples :
    python pipeline.py --agents=SAM,LISA --agents_args="SAM=--task=background --input=dog.jpeg --model-type=vit_b --bbox=[65,250,631,940]" --agents_args "LISA=--input_img=do  --input_prompt 'There is a defect ? Explain in details.' --version='xinlai/LISA-13B-llama2-v1-explanatory'" --agents_map "LISA-SAM=lisa_img_in=img_without_bg" --savefile --logfile

===========
Exemples :
python pipeline.py --agents=SAM,SAM --agents_args "SAM=--task=background --input=dog.jpeg --model-type=vit_b --bbox=[65,250,631,900]" "SAM=--task=background --input=do --model-type=vit_b" --agents_map "SAM=" "SAM-SAM=sam_img_in=img_without_bg" --savefile --logfile

python pipeline.py --agents=SAM,LISA --agents_args="SAM=--task=background --input=dog.jpeg --model-type=vit_b --bbox=[65,250,631,940]" --agents_args "LISA=--input_img=do  --input_prompt 'There is a defect ? Explain in details.' --version='xinlai/LISA-13B-llama2-v1-explanatory'" --agents_map "LISA-SAM=lisa_img_in=img_without_bg" --savefile --logfile


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
import importlib
import logging
import os
import shlex
import sys
import time
from typing import Final, Literal

# /* Modules externes */
import numpy as np
from pprint import pprint

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
        import pipeline as TP
        a=TP.PipelineLogger()  # =>Seulement stdout
        a=TP.PipelineLogger(filepath='tt.txt')  # => sdtout + fichier
        a=TP.PipelineLogger(filepath='tt.txt', is_print=False)  # =>Seulement fichier
        a("test")
        a("erreur", level=logging.ERROR)
        a.setLevel(logging.INFO, "console")
        a.setLevel(logging.WARNING, "file")
        a("erreur", level=logging.INFO)  # Affiche en console et pas dans le fichier
        # CAS : automatique
        import pipeline as TP
        args = {"nolog":False, "logfile":sys.stdout, "log_is_print":True, "log_folder":"log/timestamp/"}
        logger = TP.get_logger(args)
        ```
    """

    def __init__(self, filepath: str | None = None, is_print: bool = True,
                 logger_name: str = __name__):
        """Initialise la classe et le logger.

        :param (str) filepath:    fichier de log
        :param (bool) is_print:   indique si les logs s'affiche aussi dans stdout
        :param (str) logger_name: nom du logger à afficher dans les logs
        """
        tmp_txt = ""
        self.file = sys.stdout
        if filepath is not None:
            self.file = filepath

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
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
            tmp_txt = f"et dans {self.file}"

        self.logger = logger
        self(f"Loggeur à stdout={is_print} {tmp_txt}.")

    def __call__(self, msg: str, level: int | str = logging.INFO, caller_name: str = "", **kwargs):
        """Permet l'appel et wrapper sur la classe `logging.Logger.`

        :param (str) msg:   message à logger
        :param (int | str): [défaut INFO] niveau de log (cf. logging)
        :param (str) caller_name:   nom du logger à afficher dans les logs
        """
        log_fct = {
            logging.DEBUG: self.logger.debug,
            "debug": self.logger.debug,
            logging.INFO: self.logger.info,
            "info": self.logger.info,
            logging.WARNING: self.logger.warning,
            "warning": self.logger.warning,
            logging.ERROR: self.logger.error,
            "error": self.logger.error,
            logging.CRITICAL: self.logger.critical,
            "critical": self.logger.critical,
        }
        if isinstance(level, str):
            level = level.lower()
        if caller_name == "":
            caller_name = self.__class__.__name__
        log_fct.get(level, logging.INFO)(f"{caller_name} - {msg}", **kwargs)

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

    def get_logger_for(self, caller_name):
        """Récupère un logger pré-configurer avec le nom."""
        return LoggerProxy(self, caller_name)


class LoggerProxy:
    """Classe wrapper sur PipelineLogger afin de retenir le nom local du logger."""

    def __init__(self, logger: PipelineLogger, caller_name: str):
        self.logger: PipelineLogger = logger
        self.owner_class : str = caller_name

    def __call__(self, msg: str, level: int = logging.INFO, **kwargs):
        self.logger(msg, level=level, caller_name=self.owner_class, **kwargs)


class Pipeline:
    """Le pipeline du projet.
    ...
    """

    name: str = "pipeline"

    def __init__(self, logger: PipelineLogger | None = None):
        """Initialise le pipeline."""
        self.logger = logger
        if logger is None:  # mode console
            logger = PipelineLogger(logger_name=__name__)
        self.logger = logger.get_logger_for(self.name)
        self.logger("Initialisation création du Pipeline terminé.")

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


def find_mapping_args(args_to_map: list[str]) -> dict:
    """Trouve le mapping des arguments IN/OUT.
    
    :param (dict) item:    les paramètres fournis à mapper
    """
    # Pour chaque agent les IN et les OUT (dans agent.results)
    AGENT_MAPPING = {
        "SAM": {
            "IN": ("sam_img_in", "bbox"),
            "OUT": ("mask", "img_without_bg", "prompt")
        },
        "LISA": {
            "IN": ("lisa_img_in", "input_prompt_str", "input_expert_str"),
            "OUT": ("mask", "img_with_mask", "text_output", "prompt")
        }
    }
    results = {}

    # 1- Récupération des noms de l'agent en cours IN, et du précédent OUT
    for item in args_to_map:
        tmp_split = item.split('=', 1)
        agent_curent_name = tmp_split[0].split("-", 1)[0]
        agent_out_name = tmp_split[0].split("-", 1)[-1]
        if AGENT_MAPPING.get(agent_curent_name) is None or AGENT_MAPPING.get(agent_curent_name) is None:
            return results

        # 2- Mapping (args : IN1=OUT1,IN2=OUT2...)
        args = tmp_split[-1].split(",")
        for map_arg in args:  # (map_arg : IN1=OUT1)
            arg_in = map_arg.split("=", 1)[0]  # l'entrée du agent en cours
            arg_out = map_arg.split("=", 1)[-1]  # la sortie d'un précédent agent
            set_in = AGENT_MAPPING[agent_curent_name]["IN"]
            set_out = AGENT_MAPPING[agent_out_name]["OUT"]
            if arg_in in set_in and arg_out in set_out:
                results[agent_out_name] = results.get(agent_out_name, {})
                results[agent_out_name][arg_in] = arg_out

    return results


def log_str_format(obj) -> str:
    """Retire l'affichage des tableaux Numpy dans les logs."""
    tmp_obj = obj.copy()
    if isinstance(tmp_obj, dict):
        for item in tmp_obj:
            if isinstance(tmp_obj[item], np.ndarray):
                tmp_obj[item] = f"np.ndarray={tmp_obj[item].shape}"
    elif isinstance(tmp_obj, list):
        for i, item in enumerate(tmp_obj):
            if isinstance(tmp_obj[i], np.ndarray):
                tmp_obj[i] = f"np.ndarray={tmp_obj[i].shape}"

    return tmp_obj


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
    filename = args["logfile"] if args["logfile"] is not None else f"{Pipeline.name}.log"
    timestamp = str(int(time.time()))
    foldername = os.path.join(DEFAULT_LOG_FOLDER, timestamp)
    if args["nolog"] or not isinstance(filename, str):
        args["log_filepath"] = None
    else:
        folder = args.get("log_folder", foldername)
        args["log_filepath"] = os.path.join(folder, filename)
    # log_is_print = (bool) local pour afficher les log
    args["log_is_print"] = args.get("log_is_print", not args["nolog"])

    is_saving = False
    args["save_filepath"] = args.get("save_filepath", None)
    args["save_is_print"] = args.get("save_is_print", False)
    args["save_folder"] = args.get("save_folder", os.path.join(DEFAULT_SAVE_FOLDER, timestamp))
    if args["savefile"] is None or isinstance(args["savefile"], str):  # SAVE :
        is_saving = True
        filename = args["savefile"]
        foldername = args["save_folder"]
        if filename is not None:
            args["save_filepath"] = os.path.join(foldername, filename)
        else:
            args["save_is_print"] = True
    args["results_save_folder"] = args["save_folder"]

    args_g = {}
    args_g_name = ["nolog", "logfile", "savefile", "log_filepath", "log_is_print",
                   "save_is_print", "save_filepath", "save_folder", "results_save_folder"]
    for arg in args_g_name:
        args_g[arg] = args[arg]

    args["agents"] = args.get("agents", [])
    args["agents_args"] = args.get("agents_args", [])
    args["agents_map"] = args.get("agents_map", [])

    ###
    # Gestion du flux d'exécution
    ###
    # return
    # 0 - Création du logger
    logger = get_logger(args)

    # 1 - Création du pipeline
    pipeline = Pipeline(logger)

    if args.get("test_sam", False):
        pipeline.test_sam_lisa()
        return pipeline

    pipeline.logger(f"Arguments du Pipeline reçus : {args=}")
    module_mapping = {"SAM": "ia_sam", "LISA": "ia_lisa", "AGENT": "agentIA"}
    agents = []
    pipeline.logger(f"Début exécution dynamique des agents.")
    for i in range(len(args["agents"])):
        agent_name = args["agents"][i]
        agent_name_count = [x["agent_name"] for x in agents].count(agent_name)

        inputs = {}
        module_name = module_mapping.get(agent_name, module_mapping["AGENT"])
        module_agent = importlib.import_module(module_name)
        agent_args = {}
        if len(args["agents_args"]) > i:
            agent_args = vars(module_agent.parse_args(args["agents_args"][i]))
        agent_args.update(args_g)

        # Mapping des IN/OUT
        if i > 0:
            agent_map = {}
            agent_map_arg = {}
            if len(args["agents_map"]) > i:
                agent_map = find_mapping_args(args["agents_map"][i].split(","))
            # Simple N-1 uniquement
            # agent_map=find_mapping_args(["LISA-SAM=lisa_img_in=img_without_bg", "SAM-LISA="])
            # {'SAM': {'lisa_img_in': 'img_without_bg'}, 'LISA': {}}
            post_agents = {i:args["agents"][idx] for idx in range(i,-1,-1)}
            post_agent = agents[-1]["agent"]  # SAM précédent
            if agent_map != {}:
                args_map = list(agent_map.values())[0]  # le 1er agent à mapper (SAM)
                for arg_in in args_map:
                    agent_map_arg[arg_in] = post_agent.results[args_map[arg_in]]  # SAM.results["img_without_bg"]

            agent_args.update(agent_map_arg)

        agent_args["agent_add_name"] = ""
        agent_args["agent_add_name"] = f"_{agent_name_count}"
        pipeline.logger(f"Exécution AGENT{i}={agent_name}({agent_name}{agent_args["agent_add_name"]}) avec {log_str_format(agent_args)=}.")
        agent = None
        agent = module_agent.run_process(agent_args, logger)
        agents.append({"agent": agent, "agent_args": agent_args, "agent_name": agent_name})
        pipeline.logger(f"Fin d'exécution AGENT{i}={agent_name}({agent_name}{agent_args["agent_add_name"]})")

    return pipeline


###############################################################################
# FONCTIONS MODE CONSOLE :
def parse_args(args_str: str | None = None) -> argparse.Namespace:
    """Gestion des arguments à modifier en fonction de l'agent IA.

    ====
    Arguments :
        logfile [--logfile [FILENAME]] = défaut stdout, None si pas FILENAME
        nolog [--nolog] = (bool)
        agents [--agents AGENT1,[AGENT2,...]] = liste ordonnée des agents
        agents_args [--agents_args AGENT1="ARGS" [AGENT2="ARGS" ...]] = liste des arguments aux agents.
        agents_map [--agents_map AGENT1-AGENT2="MAPPING"] = liste des mappings IN/OUT des agents

    :param (str) args_str: pour simuler les arguments données au programme
    :return (argparse.Namespace):   les arguments parsés
    """
    if args_str is not None:
        args_str = shlex.split(args_str)

    # 1 - Définition des listes de choix :
    list_task_agentIA = ["run"]

    # 2 - Création du parseur à arguments:
    parser = argparse.ArgumentParser(prog="Pipeline",
                                     description="Gère les différents enchaînement des agents.",
                                     epilog="""Exemples :
    python pipeline.py --agents=SAM,LISA --agents_args="--task=background --input=dog.jpeg --model-type=vit_b --bbox=[65,250,631,940]" --agents_args "LISA==--input_img=LISA/1466_2L1_cut.jpg  --input_prompt 'There is a defect ? Explain in details.' --version='xinlai/LISA-13B-llama2-v1-explanatory'"
                                     """,
                                     formatter_class=argparse.RawTextHelpFormatter)

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
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_MODEL_FOLDER,
                        help="[défaut=checkpoints] dossier où sont les poids des modèles.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="[défaut=cpu] device où charger le modèle [auto, cpu, cuda, torch_directml]")

    # 3.2 args spécifique
    parser.add_argument("--test_sam", action='store_true')
    parser.add_argument("--agents", default=[], action="append",
                        help="Liste ordonnée d'agent à exécuter. (agent1,agent2)")
    parser.add_argument("--agents_args", nargs='+', default=[], action="append",
                        help="""Liste ordonnée des args pour chaque agent en forme dict (comme en ligne de commande).
    "AGENT_NAME=<args>;AGENT_NAME=<args>;..."
    --> "<args> : arg_name:value,arg_name_whitout,..."
                        """)
    parser.add_argument("--agents_map", nargs='+', default=[], action="append",
                        help="Liste ordonnée d'agent à exécuter. (agent1,agent2)")

    # 4 - Parser les arguments
    args = parser.parse_args(args_str)

    # 5 - Flatten agents (si "--agents=SAM,LISA")
    agents = []
    for entry in args.agents:
        agents.extend(entry.split(','))
    # end for
    args.agents = [a.strip().upper() for a in agents]

    if len(args.agents_args) == 1 and isinstance(args.agents_args[0], list):
        args.agents_args = args.agents_args[0]
    if len(args.agents_map) == 1 and isinstance(args.agents_map[0], list):
        args.agents_map = args.agents_map[0]

    for i, agent_name in enumerate(args.agents):
        if len(args.agents_args) <= i:
            args.agents_args.insert(i, "")

        if len(args.agents_map) <= i:
            args.agents_map.insert(i, "")

        item = args.agents_args[i]
        tmp_split = ["", ""]
        if len(item) > 1:
            tmp_split = item.split('=', 1)

        if tmp_split[0] == agent_name:
            args.agents_args[i] = tmp_split[-1]
        else:
            if tmp_split[0] != "":
                args.agents_args.insert(i, "")

        item = args.agents_map[i]
        tmp_split = ["", ""]
        if len(item) > 1:
            tmp_split = item.split('=', 1)

        if tmp_split[0].split("-", 1)[0] == agent_name:
            args.agents_map[i] = item
        elif tmp_split[0] != "":
            args.agents_map.insert(i, "")

    return args


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
