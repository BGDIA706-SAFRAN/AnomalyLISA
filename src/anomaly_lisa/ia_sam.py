"""BGDIA706 Projet fil-rouge, SAFRAN.

Fichier ia_sam.py : Défini l'agent SAM

Il fonctionne en ligne de commande ou en appel :
    - ia_sam.run_process(**args**, [logger])
    - python **args**

Où **args** sont a minima :
    python ia_sam.py -h
    ['ia_sam.py', '-h']
    usage: AgentIA SAM [-h] [--logfile [LOGFILE]] [--savefile [SAVEFILE]] [--task {run,train,background}] [--nolog]
                       [--bbox BBOX] --input INPUT [--output [OUTPUT]] [--model-type {vit_h,vit_l,vit_b}]
                       [--checkpoint CHECKPOINT] [--device DEVICE]

    command line SAM

    options:
      -h, --help            show this help message and exit
      --logfile [LOGFILE]   sortie du programme
      --savefile [SAVEFILE]
                            spécifie s'il faut sauvegarder. Si aucun fichier, alors nom par défaut
      --task {run,train,background}
                            [défaut=run] tâche à accomplir par l'agent
      --nolog               désactive les log
      --bbox BBOX           coordonnées de la box au format [x_min, y_min, x_max, y_max]
      --input INPUT         chemin de l'image d'entrée
      --output [OUTPUT]     [défaut=results] chemin du dossier de sortie
      --model-type {vit_h,vit_l,vit_b} =>"model_type" dans la variable **args**
                            [défaut=vit_h] modèle VIT de SAM [vit_h, vit_l, vit_b]
      --checkpoint CHECKPOINT
                            [défaut=checkpoints] dossier où sont les poids de SAM
      --device DEVICE       [défaut=cpu] device où charger SAM [auto, cpu, cuda, torch_directml]

Et en mode exécution Python des args optionnels supplémentaires sont :
 - pour la fonction Agent_SAM.save :
    --> results_save_filename
    --> results_save_folder
 - pour la fonction Agent_SAM.run
    --> multimask_output
    --> sam_img_in (auto-créé) avec run_process())

===========
Exemples :
python ia_sam.py --input=dog.jpeg --model-type=vit_b --bbox=[65,250,631,940] --task run --savefile
    =>exécute SAM en mode box et sauvegarde le résultat mask (nom par défaut)

python ia_sam.py --input=dog.jpeg --model-type=vit_b --bbox=[65,250,631,940] --task background --savefile
    =>fait le découpage du fond et sauvegarde les résultats mask+img_without_bg (noms par défaut)

"""
__author__ = ['Nicolas Allègre', 'Sarah Garcia', 'Luca Hachani', 'François-Xavier Morel']
__date__ = '17/05/2025'
__version__ = '0.2'

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
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

# /* Module interne */
import pipeline
from agentIA import AgentIA
from pipeline import PipelineLogger
IA_AGENT_SAM_FOLDER = "SAM"
sys.path.append(IA_AGENT_SAM_FOLDER)
from SAM.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

###############################################################################
# CONSTANTES :
EXIT_OK: Final[int] = 0

DEFAULT_SAVE_MODEL_FILENAME: Final[str] = "sam_local.pth"
DEFAULT_SAVE_RESULT_FILENAME: Final[str] = "SAM"
DEFAULT_SAM_MODEL: Final[str] = "vit_h"

DEVICE_GLOBAL: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# CLASSES PROJET :
class Agent_SAM(AgentIA):
    """Classe de l'agent SAM

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

    name: str = "SAM"
    SAM_URL: dict[str, str] = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",  # 2.39G
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",  # 1.2G
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",  # 358M
    }

    def __init__(self, logger: PipelineLogger | None = None,
                 model_type: str = DEFAULT_SAM_MODEL,
                 checkpoint_path: str = pipeline.DEFAULT_MODEL_FOLDER,
                 device=DEVICE_GLOBAL):
        """Initialise cette classe générique."""
        super().__init__(logger)

        self.model_type = model_type
        self.device = device
        self.model = self.load(self.model_type, checkpoint_path)
        self.model = self.model.to(self.device)

    def load(self, model_type: str = DEFAULT_SAM_MODEL,
                 model_dir: str = pipeline.DEFAULT_MODEL_FOLDER) -> torch.nn.Module:
        """Charge SAM soit à partir du disque soit en le téléchargeant.

        Utilise un hack en utilisant une fonction de Torch (`torch.hub.load_state_dict_from_url`)
        pour déléguer la bonne présence des poids sinon de les télécharger. Mais
        comme cela s'occupe uniquement des poids, le modèle n'est pas construit.
        D'où la suppression en mémoire des poids avant de charger SAM correctement,
        afin d'éviter d'avoir les poids 2 fois en mémoire.

        :param (str) model_type:    le type de model SAM
        :param (str) model_dir:     le dossier où sont les poids
            si le dossier ou les poids n'existent pas, ils sont téléchargés dedans
        :return (segment_anything.modeling.sam.Sam):    le modèle SAM demandé
        """
        # 1-Vérification présence des poids sinon téléchargement
        url = self.SAM_URL.get(model_type, self.SAM_URL[DEFAULT_SAM_MODEL])
        # model = torch.utils.model_zoo.load_url(url, model_dir=model_dir, weights_only=True)
        model = torch.hub.load_state_dict_from_url(url, model_dir=model_dir, weights_only=True)
        del model

        # 2-Création et chargement de SAM en mémoire demandée
        checkpoint_path = os.path.join(model_dir, os.path.basename(url))
        self.logger(f"Chargement SAM de {checkpoint_path}")
        model = sam_model_registry[model_type](checkpoint=checkpoint_path)

        return model

    def run(self, args: dict | None = None, mode="mask"):
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

        image_rgb = args.get("sam_img_in")
        if image_rgb is None:
            self.logger("image non trouvée !", level=pipeline.logging.WARNING)
            return results
        bbox = args.get("bbox")
        if bbox is None:
            h, w, _ = image_rgb.shape
            bbox = [0, 0, w, h]  # [x1, y1, x2, y2]
        box_np = np.array(ast.literal_eval(bbox))
        # TODO:Vérifier box in image ... 
        multimask_output = args.get("multimask_output", False)

        # 1 - Prédiction SAM
        self.logger(f"Prédiction : {box_np}")
        mask_predictor = SamPredictor(self.model)
        mask_predictor.set_image(image_rgb)
        masks, scores, logits = mask_predictor.predict(
            box=box_np,
            multimask_output=multimask_output,
        )
        mask = masks[0]  # shape: (H, W), bool

        # 2 - Exécution
        if mode == "background":
            self.logger("Exécution en mode background :")
            x_min, y_min, x_max, y_max = box_np
            img_without_bg = self.create_img_whitout_bg(mask, image_rgb, box_np)
            results["img_without_bg"] = img_without_bg[y_min:y_max, x_min:x_max]
        else:
            self.logger(f"Exécution non implémenté {mode}", level=pipeline.logging.WARNING)

        results["mask"] = mask
        self.results = results

    def create_img_whitout_bg(self, mask, image_rgb, box_np):
        x_min, y_min, x_max, y_max = box_np
        mask = mask[:, :, None]

        # Determine the background color to max contrast
        tone = (image_rgb * mask)[y_min:y_max, x_min:x_max].mean()
        background_color = 0 if tone > 255/2 else 255

        # Create croped image, freed from its original background
        background = np.ones(image_rgb.shape)*background_color
        final_img = (1-mask)*background + mask*image_rgb
        final_img = final_img.clip(0, 255).astype(np.uint8)

        return final_img

    # Rappel classe mère AgentIA :
    # def train(self)
    # def save(self, mode: Literal["all", "model", "results"] = "all", args: dict | None = None)
    # def save_model(self, args: dict | None = None)
    # def save_results(self, args: dict | None = None)
    def save_results(self, args: dict | None = None):
        """Sauvegarde les résultats.

        :param (dict) args: les arguments pour la sauvegarde du modèle
            args["save_filepath"]
            args["results_save_folder"]
        """
        self.logger("save")
        if args is None:
            args = {}
        filename = args.get("results_save_filename", DEFAULT_SAVE_RESULT_FILENAME)
        foldername = args.get("results_save_folder", pipeline.DEFAULT_SAVE_FOLDER)
        img_mask_filename = f"{filename}_mask.jpg"
        img_bg_filename = f"{filename}_without_bg.jpg"

        img_to_save = []
        if self.results.get("mask") is not None:
            img_mask_PIL = Image.fromarray(self.results["mask"])
            img_to_save.append((img_mask_PIL, img_mask_filename))

        if self.results.get("img_without_bg") is not None:
            img_bg_PIL = Image.fromarray(self.results["img_without_bg"])
            img_to_save.append((img_bg_PIL, img_bg_filename))

        for img, filename in img_to_save:
            filepath = os.path.join(foldername, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.logger(f"Sauvegarde des résultats : {filepath}")
            img.save(filepath)


###############################################################################
# CLASSES PROJET :
def run_process(args: dict | None = None, logger: PipelineLogger | None = None) -> Agent_SAM:
    """Exécute le déroulé d'une tâche.

    Déroule le fonctionnement de l'agent SAM.
        run :           exécute SAM pour ses fonctions intrasecs
        background :    exécution SAM pour retirer le background d'une image

    :param (dict) args:    les paramètres fournis
    :return (Agent_SAM):     l'agent utilisé avec ses résultats et modèle
    """
    ###
    # Gestion des arguments par défaut - compatible mode console ou mode Python
    ###
    if args is None:
        args = {}

    #3.1 args génériques
    # task [--task TASK] = (str) "run", "train", ...
    args["task"] = args.get("task", "run")
    # nolog [--nolog] = (bool)
    args["nolog"] = args.get("nolog", False)
    # logfile [--logfile [FILENAME]] = défaut stdout, None si pas FILENAME
    args["logfile"] = args.get("logfile", sys.stdout)
    # savefile [--savefile [FILENAME]] = défaut stdout, None si pas FILENAME
    args["savefile"] = args.get("savefile", sys.stdout)
    # log_filepath = (str) None si pas de log en fichier
    filename = args["logfile"] if args["logfile"] is not None else f"{Agent_SAM.name}.log"
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

    # 3.2 args spécifique SAM
    # bbox [--bbox ["[x1,y1,x2,y2]"]] = (str(list)) = défaut None si non indiqué
    args["bbox"] = args.get("bbox", None)
    # input --input FILENAME = (str) chemin de l'image
    args["input"] = args.get("input", "image.jpg")  # défaut ??
    # output [--output [FOLDER]] = (str) défaut 'results'
    args["output"] = args.get("output", pipeline.DEFAULT_SAVE_FOLDER)
    # model_type [--model-type VIT_TYPE] = (str) "vit_h", "vit_l", "vit_b"
    args["model_type"] = args.get("model_type", DEFAULT_SAM_MODEL)
    # checkpoint [--checkpoint FOLDER] = (str)
    args["checkpoint"] = args.get("checkpoint", pipeline.DEFAULT_MODEL_FOLDER)
    # device [--device DEVICE] = (str) 'auto, cpu, cuda, torch_directml'
    # args["device"] = args.get("device", utils.torch_pick_device())
    args["device"] = args.get("device", DEVICE_GLOBAL)

    ###
    # Gestion du flux d'exécution
    ###
    pprint(args)  # pour test
    # 0 - Création du logger
    if logger is None:
        # logger = pipeline.get_logger(args)
        logger = PipelineLogger(filepath=args["log_filepath"], is_print=args["log_is_print"], logger_name=Agent_SAM.name)

    # 1- Création et configuration agent
    # agent = Agent_SAM(logger=logger, model_type="vit_b")
    agent = Agent_SAM(logger, model_type=args["model_type"], checkpoint_path=args["checkpoint"], device=args["device"])

    # # 2- Suivant la tâche exécution de celle-ci
    image_filename = args["input"]
    image_PIL = Image.open(image_filename)
    image_PIL = image_PIL.convert("RGB")
    image_rgb = np.array(image_PIL)
    local_arg = {}
    if args["task"] == "run":
        local_arg = {
            "print": True,  # local print sans le logger
            "sam_img_in": image_rgb,
            "bbox": args["bbox"],
        }
        agent.logger("Action : exécution SAM courante ...")
        agent.run(local_arg, mode="mask")
    elif args["task"] == "train":
        agent.logger("Action : entraînement ...")
        agent.train()
    elif args["task"] == "background":
        local_arg = {
            "print": True,  # local print sans le logger
            "sam_img_in": image_rgb,
            "bbox": args["bbox"],
        }
        agent.logger("Action : enlever le fond ...")
        agent.run(local_arg, mode="background")

    # # 3-Sauvegarde
    modes: dict[str, str] = {"run": "results", "train": "model", "background": "results"}
    mode: str = modes.get(args["task"], "results")

    local_arg["save_filepath"] = args["save_filepath"]
    local_arg["save_is_print"] = args["save_is_print"]
    if is_saving:
        agent.logger("Action : saving ...")
        agent.save(mode, local_arg)

    return agent


###############################################################################
# FONCTIONS MODE CONSOLE :
def parse_args() -> argparse.Namespace:
    """Gestion des arguments de l'agent SAM.

    ====
    Arguments :
        task [--task TASK] = (str) "run", "train", "background"
        logfile [--logfile [FILENAME]] = défaut stdout, None si pas FILENAME
        savefile [--savefile [FILENAME]] = défaut stdout, None si pas FILENAME
        nolog [--nolog] = (bool) désactive les logs
        bbox [--bbox ["[x1,y1,x2,y2]"]] = (str(list)) = défaut None si non indiqué
        input --input FILENAME = (str) chemin de l'image
        output [--output [FOLDER]] = (str) défaut 'results'
        model_type [--model-type VIT_TYPE] = (str) "vit_h", "vit_l", "vit_b"
        checkpoint [--checkpoint FOLDER] = (str)
        device [--device DEVICE] = (str) 'auto, cpu, cuda, torch_directml'

    :param (str) args: les arguments données au programme
    :return (argparse.Namespace):   les arguments parsés
    """
    # 1 - Définition des listes de choix :
    list_task_agentIA = ["run", "train", "background"]
    list_model_SAM = ["vit_h", "vit_l", "vit_b"]

    # 2 - Création du parseur à arguments:
    parser = argparse.ArgumentParser(prog="AgentIA SAM",
                                     description="command line SAM")

    # 3 - Définition des arguments :
    #3.1 args génériques
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
                        help="[défaut=checkpoints] dossier où sont les poids de SAM")
    parser.add_argument("--device", type=str, default="cpu",
                        help="[défaut=cpu] device où charger SAM [auto, cpu, cuda, torch_directml]")

    # 3.2 args spécifique SAM
    parser.add_argument("--bbox", type=str, default=None,
                        help='coordonnées de la box au format [x_min, y_min, x_max, y_max]')                    
    parser.add_argument("--input", type=str, required=True,
                        help="chemin de l'image d'entrée")
    parser.add_argument("--output", type=str, nargs='?', default=pipeline.DEFAULT_SAVE_FOLDER,
                        help="[défaut=results] chemin du dossier de sortie")
    parser.add_argument("--model-type", type=str, choices=list_model_SAM, default=DEFAULT_SAM_MODEL,
                        help="[défaut=vit_h] modèle VIT de SAM [vit_h, vit_l, vit_b]")

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

# import importlib
# import ia_sam
# args={"model-type":"vit_b", "input":"dog.jpeg", "bbox":"[65,250,631,940]"}
# agent = ia_sam.run_process(args)
