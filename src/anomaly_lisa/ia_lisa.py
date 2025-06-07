"""BGDIA706 Projet fil-rouge, SAFRAN.

Fichier ia_lisa.py : Défini l'agent LISA

Il fonctionne en ligne de commande ou en appel :
    - ia_lisa.run_process(**args**, [logger])
    - python **args**

Où **args** sont a minima :
    python ia_lisa.py -h
    usage: AgentIA SAM [-h] [--logfile [LOGFILE]] [--savefile [SAVEFILE]] [--task {run,train,background}] [--nolog]
                       [--output [OUTPUT]] [--checkpoint CHECKPOINT] [--device DEVICE] --input_img INPUT_IMG
                       --input_prompt INPUT_PROMPT [--input_expert INPUT_EXPERT]
                       [--version {xinlai/LISA-7B-v1-explanatory,xinlai/LISA-13B-llama2-v1-explanatory,Senqiao/LISA_Plus_7b}]
                       [--precision {fp32,bf16,fp16}] [--image_size IMAGE_SIZE] [--load_in_8bit] [--load_in_4bit]

    command line LISA

    options:
      -h, --help            show this help message and exit
      --logfile [LOGFILE]   sortie du programme
      --savefile [SAVEFILE]
                            spécifie s'il faut sauvegarder. Si aucun fichier, alors stdout sauf pour 'train'
      --task {run,train,background}
                            [défaut=run] tâche à accomplir par l'agent
      --nolog               désactive les log
      --output [OUTPUT]     [défaut=results] chemin du dossier de sortie
      --checkpoint CHECKPOINT
                            [défaut=checkpoints] dossier où sont les poids du modèle
      --device DEVICE       [défaut=cpu] device où charger SAM [auto, cpu, cuda, torch_directml]
      --input_img INPUT_IMG
                            chemin de l'image d'entrée
      --input_prompt INPUT_PROMPT
                            chemin du fichier prompt d'entrée
      --input_expert INPUT_EXPERT
                            chemin du fichier prompt des experts
      --version {xinlai/LISA-7B-v1-explanatory,xinlai/LISA-13B-llama2-v1-explanatory,Senqiao/LISA_Plus_7b}
                            [défaut=LISA7B] dépôt du modèle
      --precision {fp32,bf16,fp16}
                            precision for inference
      --image_size IMAGE_SIZE
                            image size
      --load_in_8bit
      --load_in_4bit


Et en mode exécution Python des args optionnels supplémentaires sont :
 - pour la fonction Agent_LISA.save :
    --> results_save_filename
    --> results_save_folder
 - pour la fonction Agent_LISA.run
    --> lisa_img_in (auto-créé) avec run_process())

===========
Exemples :
CUDA_VISIBLE_DEVICES=1 python ia_lisa.py --input_img=LISA/1466_2L1_cut.jpg  --input_prompt "There is a defect ? Explain in details." --device cuda --savefile

CUDA_VISIBLE_DEVICES=1 python ia_lisa.py --input_img=../data/MMAD/MVTec-AD/carpet/test/cut/000.png --input_prompt ./user_prompt.txt --input_expert ./expert_prompt.txt --device cuda --savefile

CUDA_VISIBLE_DEVICES=2 python ia_lisa.py --input_img=../data/MMAD/MVTec-AD/bottle/test/broken_small/004.png  --input_prompt "According to experts, it is a bottle seen from above and there is a defect to the bottom left. Is there any defect in the object? if yes describe it and explain and give a segmentation mask." --device cuda --savefile --version='xinlai/LISA-13B-llama2-v1-explanatory' --type_prompt=v0

Exemple dans une console Python :
import ia_lisa
agent = ia_lisa.Agent_LISA()  # chargement par défaut
agent.run({"lisa_img_in":'../data/MMAD/MVTec-AD/bottle/test/broken_small/004.png'})
agent.results.keys()
# >> dict_keys(['text_output', 'pred_masks', 'image_np'])

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
import random
import sys
import time
from pprint import pprint
from typing import Any, Final, Literal

# /* Modules externes */
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

# /* Module interne */
import pipeline
from agentIA import AgentIA
from pipeline import PipelineLogger
IA_AGENT_LISA_FOLDER = "LISA"
sys.path.append(IA_AGENT_LISA_FOLDER)
from LISA.model.LISA import LISAForCausalLM
from LISA.model.llava import conversation as conversation_lib
from LISA.model.llava.mm_utils import tokenizer_image_token
from LISA.model.segment_anything.utils.transforms import ResizeLongestSide
from LISA.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                              DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                              EXPLANATORY_QUESTION_LIST)


###############################################################################
# CONSTANTES :
EXIT_OK: Final[int] = 0

DEFAULT_SAVE_MODEL_FILENAME: Final[str] = "lisa_local.pth"
DEFAULT_SAVE_RESULT_FILENAME: Final[str] = "LISA"
DEFAULT_LISA_MODEL: Final[str] = "xinlai/LISA-7B-v1-explanatory"

DEFAULT_LISA_MODEL_IMAGE_SIZE = 1024
DEFAULT_LISA_MODEL_MAX_LENGTH = 512
DEFAULT_LISA_MODEL_LORA = 8
DEFAULT_LISA_MODEL_VISION_TOWER = "openai/clip-vit-large-patch14"
DEFAULT_LISA_MODEL_USE_MM_START_END = True
DEFAULT_LISA_MODEL_CONV_TYPE = "llava_v1"

DEVICE_GLOBAL: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# CLASSES PROJET :
class Agent_LISA(AgentIA):
    """Classe de l'agent LISA

    =============
    Attributs (mère) :
        name      (class) le nom de l'agent
        models    l'IA de l'agent IA
        results   les résultats de l'exécution
        logger    le logger pour les messages

    =============
    Méthodes
        run(args: dict())   Exécute la tâche de l'agent IA.
        train()             Entraînement le modèle de l'agent IA.
        save(mode, args)    Sauvegarde du modèle entraîné ou les résultats.

    =============
    Exemples :
        import ia_lisa
        agent = ia_lisa.Agent_LISA()  # chargement par défaut
        agent.run({"lisa_img_in":'../data/MMAD/MVTec-AD/bottle/test/broken_small/004.png'})
        agent.results.keys()
        # >> dict_keys(['text_output', 'pred_masks', 'image_np'])
    """

    name: str = "LISA"

    def __init__(self, args: dict | None = None,
                 logger: PipelineLogger | None = None, logger_name: str = ""):
        """Initialise cette classe."""
        super().__init__(logger, logger_name)

        self.models: dict = self.load(args)

        self.tokenizer = self.models["tokenizer"]
        self.model = self.models["model"]
        self.clip_image_processor = self.models["clip_image_processor"]
        self.transform = self.models["transform"]

    def load(self, args: dict) -> dict:
        """Charge en mémoire le modèle LISA.

        :param (dict) args: les arguments nécessaire
            args["version"]
            args["model_max_length"]
            args["precision"]
            args["load_in_4bit"]
            args["load_in_8bit"]
            args["vision_tower"]
            args["image_size"]
        :return (dict): les éléments du modèle
            models["tokenizer"] = tokenizer
            models["model"] = model
            models["clip_image_processor"] = clip_image_processor
            models["transform"] = transform
        """
        models: dict[str, Any] = {}

        # 0-Gestion des arguments
        if args is None:
            args = {}
        device = args.get("device", DEVICE_GLOBAL)
        args["version"] = args.get("version", DEFAULT_LISA_MODEL)
        args["precision"] = args.get("precision", "bf16")
        args["image_size"] = args.get("image_size", DEFAULT_LISA_MODEL_IMAGE_SIZE)
        args["load_in_8bit"] = args.get("load_in_8bit", False)
        args["load_in_4bit"] = args.get("load_in_4bit", False)
        args["model_max_length"] = args.get("model_max_length", DEFAULT_LISA_MODEL_MAX_LENGTH)
        args["vision_tower"] = args.get("vision_tower", DEFAULT_LISA_MODEL_VISION_TOWER)

        self.logger(f"Chargement LISA de {args['version']}")
        # 1-Chargement de LISA (cf. chat.py de LISA)
        tokenizer = AutoTokenizer.from_pretrained(
            args["version"],
            cache_dir=None,
            model_max_length=args["model_max_length"],
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

        torch_dtype = torch.float32
        if args["precision"] == "bf16":
            torch_dtype = torch.bfloat16
        elif args["precision"] == "fp16":
            torch_dtype = torch.half

        kwargs = {"torch_dtype": torch_dtype}
        if args["load_in_4bit"]:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "load_in_4bit": True,
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        llm_int8_skip_modules=["visual_model"],
                    ),
                }
            )
        elif args["load_in_8bit"]:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "quantization_config": BitsAndBytesConfig(
                        llm_int8_skip_modules=["visual_model"],
                        load_in_8bit=True,
                    ),
                }
            )

        model = LISAForCausalLM.from_pretrained(
            args["version"], low_cpu_mem_usage=True,
            vision_tower=args["vision_tower"], seg_token_idx=seg_token_idx,
            **kwargs
        )
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)

        if args["precision"] == "bf16":
            model = model.bfloat16().to(device)
        elif (
            args["precision"] == "fp16" and (not args["load_in_4bit"]) and (not args["load_in_8bit"])
        ):
            vision_tower = model.get_model().get_vision_tower()
            model.model.vision_tower = None
            import deepspeed

            model_engine = deepspeed.init_inference(
                model=model,
                dtype=torch.half,
                replace_with_kernel_inject=True,
                replace_method="auto",
            )
            model = model_engine.module
            model.model.vision_tower = vision_tower.half().to(device)
        elif args["precision"] == "fp32":
            model = model.float().to(device)

        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(device)

        clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
        transform = ResizeLongestSide(args["image_size"])

        model.eval()

        self.logger(f"Chargement LISA : FIN")
        models["tokenizer"] = tokenizer
        models["model"] = model
        models["clip_image_processor"] = clip_image_processor
        models["transform"] = transform

        return models

    def run(self, args: dict | None = None):
        """Exécute la tâche de l'agent IA et enregistre les résultats dans self.results.

        :param (dict) args: les arguments nécessaire
            args["conv_type"]
            args["type_prompt"]
            args["input_prompt_str"]
            args["input_expert_str"]
            args["use_mm_start_end"]
            args["lisa_img_in"]  np.array() ou Path_to_img
            args["precision"]
            args["print"] affiche des infos (if args.get("print", False): #print)
        """
        results: dict[str, Any] = {}

        # 0-Gestion des arguments
        if args is None:
            args = {}
        is_print = args.get("print", False)
        if is_print:
            print("Exécution de ", self.name)

        device = args.get("device", DEVICE_GLOBAL)
        args["use_mm_start_end"] = args.get("use_mm_start_end", DEFAULT_LISA_MODEL_USE_MM_START_END)
        args["conv_type"] = args.get("conv_type", DEFAULT_LISA_MODEL_CONV_TYPE)
        args["precision"] = args.get("precision", "bf16")
        args["type_prompt"] = args.get("type_prompt", "v0")
        args["input_expert_str"] = args.get("input_expert_str", "")
        if args.get("input_prompt_str") is None:
            args["input_prompt_str"] = " {}".format(random.choice(EXPLANATORY_QUESTION_LIST))

        image_rgb = args.get("lisa_img_in")
        if image_rgb is None:
            self.logger("image non donnée !", level=pipeline.logging.WARNING)
            return
        if isinstance(image_rgb, str):
            if not os.path.exists(image_rgb):
                self.logger(f"image non trouvée {image_rgb}!", level=pipeline.logging.WARNING)
                return
            self.logger(f"Chargement img {image_rgb}", level=pipeline.logging.INFO)
            image_PIL = Image.open(image_rgb)
            image_PIL = image_PIL.convert("RGB")
            image_rgb = np.array(image_PIL)

        if is_print:
            print("Exécution de ", self.name)

        # 1-Gestion du prompt
        type_prompt = args["type_prompt"]
        prompt_user = args["input_prompt_str"]
        prompt_expert = args["input_expert_str"]
        conv = conversation_lib.conv_templates[args["conv_type"]].copy()
        conv.messages = []
        # Prompt LISA chat.py (IMG + prompts)
        if type_prompt == "v0":
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_expert + "\n" + prompt_user
            if args["use_mm_start_end"]:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            conv.append_message(conv.roles[0], prompt)  # IMG+expert+user
        # Prompt FX init (IMG, expert, user)
        elif type_prompt == "v1":
            prompt = DEFAULT_IMAGE_TOKEN
            if args["use_mm_start_end"]:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            conv.append_message(conv.roles[0], prompt)  # IMG
            if len(expert_info) == 0:
                conv.append_message(conv.roles[0], prompt_expert)  # expert
            conv.append_message(conv.roles[0], prompt_user)  # user
        # Prompt LISA (IMG + expert, user)
        elif type_prompt == "v2":
            if len(prompt_expert) == 0:
                prompt_expert = "Experts are struggling to analyze and explain this image."
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_expert
            if args["use_mm_start_end"]:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            conv.append_message(conv.roles[0], prompt)  # IMG+expert
            conv.append_message(conv.roles[0], prompt_user)  # user
        # end if
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        self.logger(f"{prompt=}")

        # 2-Gestion de l'image
        image_np = image_rgb
        original_size_list = [image_np.shape[:2]]
        image_clip = (
            self.clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .to(device)
        )
        if args["precision"] == "bf16":
            image_clip = image_clip.bfloat16()
        elif args["precision"] == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()
        image = self.transform.apply_image(image_np)
        resize_list = [image.shape[:2]]
        image = (
            self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .to(device)
        )
        if args["precision"] == "bf16":
            image = image.bfloat16()
        elif args["precision"] == "fp16":
            image = image.half()
        else:
            image = image.float()

        # 3-Exécution
        start_time = time.time()
        self.logger(f"Exécution LISA")
        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(device)
        output_ids, pred_masks = self.model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=self.tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        self.logger(f"{text_output=}")
        self.logger(f"FIN exécution LISA en {time.time() - start_time:.1f} secondes")

        text_output = text_output.replace("\n", "").replace("  ", " ")
        results["text_output"] = text_output
        results["pred_masks"] = pred_masks
        results["image_np"] = image_np
        self.results = results

    def preprocess(self, x, pixel_mean=None, pixel_std=None,
                   img_size=DEFAULT_LISA_MODEL_IMAGE_SIZE) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if pixel_mean is None:
            pixel_mean = torch.Tensor([123.675, 116.28, 103.53], device=x.device).view(-1, 1, 1)
        if pixel_std is None:
            pixel_std = torch.Tensor([58.395, 57.12, 57.375], device=x.device).view(-1, 1, 1)
        # Normalize colors
        x = (x - pixel_mean) / pixel_std
        # Pad
        h, w = x.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = torch.nn.functional.pad(x, (0, padw, 0, padh))
        return x

    def create_prompt(self, args: dict | None = None):
        """Crée le prompt expert associé à la tâche.

        :param (dict) args: les arguments de la fonction
        """
        prompt: str = ""
        if len(self.results) == 0:
            self.logger(f"Pas de résultat pour créer le prompt en {mode}", level="error")
            return

        if args is None:
            args = {}
        is_print = args.get("print", False)
        device = args.get("device", DEVICE_GLOBAL)
        task = args.get("task", "run")
        if is_print:
            print(f"Création prompt pour {task} de ", self.name)

        # Faire quelque pour créer le prompt.

        self.results["prompt"] = prompt  # expert prompt pour une tâche donnée

    # Rappel classe mère AgentIA :
    # def train(self)
    # def save(self, mode: Literal["all", "model", "results"] = "all", args: dict | None = None)
    # def save_model(self, args: dict | None = None)
    def save_results(self, args: dict | None = None):
        """Sauvegarde les résultats.

        :param (dict) args: les arguments pour la sauvegarde du modèle
            args["results_save_filename"]   préfixe du nom de fichier "LISA" par défaut
            args["results_save_folder"]
        """
        if args is None:
            args = {}
        is_print = args.get("print", False)
        if is_print:
            print("Sauvegarde de ", self.name)

        type_save = args.get("type_save", "v1")
        filename = args.get("results_save_filename", self.name)
        foldername = args.get("results_save_folder", pipeline.DEFAULT_SAVE_FOLDER)
        args["results_save_filename"] = filename
        args["results_save_folder"] = foldername
        img_mask_filename = f"{filename}_mask.jpg"
        img_seg_filename = f"{filename}_seg.jpg"
        txt_filename = f"{filename}_answer.txt"
        prompt_filename = f"{filename}_prompt.txt"
        filepath = os.path.join(foldername, txt_filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # TODO : à mettre dans run !
        image_np = self.results["image_np"]
        i = 0
        pred_mask = self.results["pred_masks"][i]
        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0
        global_mask = np.uint8(pred_mask * 100)
        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[pred_mask]
        final_img = save_img
        # chat.py : cv2.imwrite(img_mask_filename, pred_mask * 100)
        # chat.py : save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        # chat.py : cv2.imwrite(img_seg_filename, save_img)

        # sauvegarde du prompt réponse
        filepath = os.path.join(foldername, txt_filename)
        with open(filepath, "w") as f:
            self.logger(f"Sauvegarde des résultats : {filepath}")
            f.write(self.results["text_output"])

        # sauvegarde des images résultats
        img_to_save = []
        img_mask_PIL = Image.fromarray(global_mask)
        img_to_save.append((img_mask_PIL, img_mask_filename))
        img_seg_PIL = Image.fromarray(final_img)
        img_to_save.append((img_seg_PIL, img_seg_filename))
        for img, filename in img_to_save:
            filepath = os.path.join(foldername, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.logger(f"Sauvegarde des résultats : {filepath}")
            img.save(filepath)

        # sauvegarde d'un prompt expert
        if self.results.get("prompt") is not None:
            filepath = os.path.join(foldername, prompt_filename)
            with open(filepath, "w") as f:
                self.logger(f"Sauvegarde du prompt expert : {filepath}")
                f.write(self.results["prompt"])


###############################################################################
# CLASSES PROJET :
def run_process(args: dict | None = None, logger: PipelineLogger | None = None) -> Agent_LISA:
    """Exécute le déroulé d'une tâche.

    Déroule le fonctionnement de l'agent LISA.
        run :           exécute LISA pour ses fonctions intrasecs
        background :    exécution LISA pour retirer le background d'une image

    :param (dict) args:    les paramètres fournis
    :return (AgentIA):     l'agent utilisé avec ses résultats et modèle
    """
    ###
    # Gestion des arguments par défaut - compatible mode console ou mode Python
    ###
    if args is None:
        args = {}

    # 3.1 args génériques
    args["agent_add_name"] = args.get("agent_add_name", "")
    # task [--task TASK] = (str) "run", "train", ...
    args["task"] = args.get("task", "run")
    # nolog [--nolog] = (bool)
    args["nolog"] = args.get("nolog", False)
    # logfile [--logfile [FILENAME]] = défaut stdout, None si pas FILENAME
    args["logfile"] = args.get("logfile", sys.stdout)
    # savefile [--savefile [FILENAME]] = défaut stdout, None si pas FILENAME
    args["savefile"] = args.get("savefile", sys.stdout)
    # log_filepath = (str) None si pas de log en fichier
    filename = args["logfile"] if args["logfile"] is not None else f"{Agent_LISA.name}.log"
    foldername = os.path.join(pipeline.DEFAULT_LOG_FOLDER, str(int(time.time())))
    if args["nolog"] or not isinstance(filename, str):
        args["log_filepath"] = None
    else:
        folder = args.get("log_folder", foldername)
        args["log_filepath"] = os.path.join(folder, filename)
    # log_is_print = (bool) local pour afficher les log
    args["log_is_print"] = args.get("log_is_print", not args["nolog"])
    # device [--device DEVICE] = (str) 'auto, cpu, cuda, torch_directml'
    # args["device"] = args.get("device", utils.torch_pick_device())
    args["device"] = args.get("device", DEVICE_GLOBAL)
    # output [--output [FOLDER]] = (str) défaut 'results'
    args["output"] = args.get("output", pipeline.DEFAULT_SAVE_FOLDER)
    # checkpoint [--checkpoint FOLDER] = (str)
    args["checkpoint"] = args.get("checkpoint", pipeline.DEFAULT_MODEL_FOLDER)

    is_saving = False
    args["results_save_folder"] = args.get("results_save_folder", pipeline.DEFAULT_SAVE_FOLDER)
    args["model_save_folder"] = args.get("model_save_folder", pipeline.DEFAULT_SAVE_FOLDER)
    args["save_filepath"] = None
    if args["savefile"] is None or isinstance(args["savefile"], str):  # SAVE :
        is_saving = True
        if isinstance(args["savefile"], str) and args.get("results_save_filename") is None:
            args["results_save_filename"] = args["savefile"]
            args["model_save_filename"] = args["savefile"]

    # 3.2 args spécifique LISA
    # version [--version VERSION] = (str)
    args["version"] = args.get("version", DEFAULT_LISA_MODEL)
    # precision [--precision PRECISION] = (str)
    args["precision"] = args.get("precision", "bf16")
    # image_size [--image_size SIZE] = (int)
    args["image_size"] = args.get("image_size", DEFAULT_LISA_MODEL_IMAGE_SIZE)
    # load_in_8bit [--load_in_8bit] = (bool)
    args["load_in_8bit"] = args.get("load_in_8bit", False)
    # load_in_4bit [--load_in_4bit] = (bool)
    args["load_in_4bit"] = args.get("load_in_4bit", False)
    # input_img --input_img FILENAME = (str) chemin de l'image
    args["input_img"] = args.get("input_img")
    # input_prompt --input_prompt FILENAME = (str) chemin de l'image
    args["input_prompt"] = args.get("input_prompt", "")
    if os.path.exists(args["input_prompt"]):
        with open(args["input_prompt"], "r") as f:
            args["input_prompt_str"] = f.read()
    else:
        args["input_prompt_str"] = args["input_prompt"]
    # input_prompt_expert --input_prompt_expert FILENAME = (str) chemin de l'image
    args["input_expert"] = args.get("input_expert", "expert_prompt.txt")  # ??
    if os.path.exists(args["input_expert"]):
        with open(args["input_expert"], "r") as f:
            args["input_expert_str"] = f.read()
    else:
        args["input_expert_str"] = ""

    # 3.3 args supplémentaires LISA
    args["model_max_length"] = args.get("model_max_length", DEFAULT_LISA_MODEL_MAX_LENGTH)
    args["lora_r"] = args.get("lora_r", DEFAULT_LISA_MODEL_LORA)
    args["vision_tower"] = args.get("vision_tower", DEFAULT_LISA_MODEL_VISION_TOWER)
    args["use_mm_start_end"] = args.get("use_mm_start_end", DEFAULT_LISA_MODEL_USE_MM_START_END)
    args["conv_type"] = args.get("conv_type", DEFAULT_LISA_MODEL_CONV_TYPE)

    ###
    # Gestion du flux d'exécution
    ###
    pprint(pipeline.log_str_format(args))  # pour test
    # 0 - Création du logger
    if logger is None:
        # logger = pipeline.get_logger(args)
        logger = PipelineLogger(filepath=args["log_filepath"], is_print=args["log_is_print"], logger_name=Agent_LISA.name)

    # 0.1 - Gestion de l'image :
    if args.get("input_img") is None and args.get("lisa_img_in") is None:
        logger("LISA a besoin d'image or aucune image n'a été fournie !", level=pipeline.logging.WARNING)
        return
    if args["lisa_img_in"] is None:
        image_filename = args["input_img"]
        image_PIL = Image.open(image_filename)
        image_PIL = image_PIL.convert("RGB")
        args["lisa_img_in"] = np.array(image_PIL)

    # 1- Création et configuration agent
    agent = Agent_LISA(args, logger=logger, logger_name=args["agent_add_name"])

    # # 2- Suivant la tâche exécution de celle-ci
    local_arg = {}
    if args["task"] == "run":
        local_arg = {
            "print": True,  # local print sans le logger
            "lisa_img_in": args["lisa_img_in"],
        }
        local_arg.update(args)
        agent.logger("Action : exécution LISA courante ...")
        agent.run(local_arg)
    elif args["task"] == "train":
        agent.logger("Action : entraînement ...")
        agent.train()

    # # 3-Sauvegarde
    modes: dict[str, str] = {"run": "results", "train": "model"}
    mode = modes.get(args["task"], "results")
    if is_saving:
        agent.logger("Action : sauvegarde ...")
        agent.save(mode, args)

    return agent


###############################################################################
# FONCTIONS MODE CONSOLE :
def parse_args(args_str: str | None = None) -> argparse.Namespace:
    """Gestion des arguments de l'agent.

    ====
    Arguments :
        task [--task TASK] = (str) "run", "train"
        logfile [--logfile [FILENAME]] = défaut stdout, None si pas FILENAME
        savefile [--savefile [FILENAME]] = défaut stdout, None si pas FILENAME
        nolog [--nolog] = (bool) désactive les logs
        output [--output [FOLDER]] = (str) défaut 'results'
        checkpoint [--checkpoint FOLDER] = (str)
        device [--device DEVICE] = (str) 'auto, cpu, cuda, torch_directml'
        input_img [--input_img [FILENAME]] = (str) image à utiliser
        input_prompt [--input_prompt [STR | FILENAME]] = prompt ou fichier contenant le prompt
        input_expert [--input_expert [STR | FILENAME]] = prompt ou fichier contenant le prompt
        version [--version MODEL_TYPE] = (str) choix du modèle LISA-7B, LISA-13B ou LISA++

    :param (str) args_str: pour simuler les arguments données au programme
    :return (argparse.Namespace):   les arguments parsés
    """
    if args_str is not None:
        args_str = args_str.split()

    # 1 - Définition des listes de choix :
    list_task_agentIA = ["run", "train"]
    list_model_LISA = ["xinlai/LISA-7B-v1-explanatory",
                       "xinlai/LISA-13B-llama2-v1-explanatory",
                       "Senqiao/LISA_Plus_7b"]
    list_precision_model = ["fp32", "bf16", "fp16"]

    # 2 - Création du parseur à arguments:
    parser = argparse.ArgumentParser(prog="AgentIA LISA",
                                     description="command line LISA")

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
    parser.add_argument("--output", type=str, nargs='?', default=pipeline.DEFAULT_SAVE_FOLDER,
                        help="[défaut=results] chemin du dossier de sortie")
    parser.add_argument("--checkpoint", type=str, default=pipeline.DEFAULT_MODEL_FOLDER,
                        help="[défaut=checkpoints] dossier où sont les poids du modèle")
    parser.add_argument("--device", type=str, default="cpu",
                        help="[défaut=cpu] device où charger le modèle [auto, cpu, cuda, torch_directml]")

    # 3.2 args spécifique LISA
    parser.add_argument("--input_img", type=str, required=True,
                        help="chemin de l'image d'entrée")
    parser.add_argument("--input_prompt", type=str, required=True,
                        help="chemin du fichier prompt d'entrée")
    parser.add_argument("--input_expert", type=str, default="",
                        help="prompt ou chemin du fichier prompt des experts")
    parser.add_argument("--version", choices=list_model_LISA,
                        default="xinlai/LISA-7B-v1-explanatory",
                        help="[défaut=LISA7B] dépôt du modèle")
    parser.add_argument("--precision", type=str, choices=list_precision_model,
                        default="bf16", help="precision for inference")
    parser.add_argument("--image_size", default=DEFAULT_LISA_MODEL_IMAGE_SIZE, type=int,
                        help="image size")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--conv_type", default=DEFAULT_LISA_MODEL_CONV_TYPE, type=str,
                        choices=["llava_v1", "llava_llama_2"],
                        help=f"[défaut={DEFAULT_LISA_MODEL_CONV_TYPE}] style de format de prompt")

    parser.add_argument("--type_prompt", default="v0", choices=["v0", "v1", "v2"],
                        help="""Format du prompt :\
                        v0 = idem chat.py même conversation, IMG + prompts
                        v1 = IMG / expert_prompt / user_prompt
                        v2 = IMG + expert_prompt / user_prompt
                        """)

    return parser.parse_args(args_str)


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
