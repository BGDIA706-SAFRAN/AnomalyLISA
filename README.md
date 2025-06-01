# **AnomalyLISA**

[![Licence EUPL 1.2](https://img.shields.io/badge/licence-EUPL_1.2-blue)](https://interoperable-europe.ec.europa.eu/collection/eupl/eupl-text-eupl-12)

AnomalyLISA : boosting a VLLM with LISA method and expert agents for Industrial Anomaly Detection


## **Description :**

L'ensemble de ce dépôt dépeint un système multi-agent pour améliorer un chatbot dans la détection d'anomalie industriel (IAD).

### **Architecture :**

L'architecture de ce système est constituée de plusieurs agents qui sont des IA experts dans leurs tâches qui sont orchestrés par un ordonnenceur.

Si la majorité des agents sont autonomes, l'ordonnenceur (pipeline) est le cerveau qui permet de faire fonctionner la globalité et de gérer les communications entre agents.

### **Type d'agent IA**

Un agent a la faculté d'exécuter une série de tâches en fonction d'informations et des consignes en entrée pour produire en sortie, des résultats et des paramètres.
- Entrée :
  - données d'entrée
  - paramètres de configuration et consignes
- Réalisation de la tâche
- Sortie :
  - résultats issus des données
  - paramètres supplémentaires issus de l'exécution de la tâche (comme des consignes d'utilisation des résultats)

## **Les agents**

Nous avons construit les différents agent qu'ils puissent êter totatelement indépendant et fonctionnel. De plus, ils peuvent s'utiliser soit en ligne de commande soit directement en Python.

### Agent SAM (`Agent_SAM`)

L'agent SAM permet d'englober le fonctionnement de SAM afin de réaliser différentes tâches nécessaire tout le long du pipeline.

**Agent SAM (pour plus de détail voir '§usage' :**
- Entrée :
  - image à traiter
  - coordonnées BBOX ou point (selon la tâche)
  - consignes et autres paramètres
- Réalisation de la tâche :
  - Tâche de pré-traitement : retrait du fond de l'image
  - Tâche de segmentation : création d'un masque sur l'objet ciblé
- Sortie (selon la tâche voulue) :
  - images
  - résultats supplémentaires de la segmentation SAM
  - prompt expert à destination d'un LLM

#### **Installation :**

```bash
git clone https://github.com/facebookresearch/segment-anything.git SAM
sed -i \"s/torch.load(f)/torch.load(f, weights_only=True)/g\" SAM/segment_anything/build_sam.py
```

#### **Usage ligne de commande :**
```
python ia_sam.py -h
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
```

- **Exemples :**
  - exécute SAM en mode box et sauvegarde le résultat mask (nom par défaut)
  ```bash
  python ia_sam.py --input=dog.jpeg --model-type=vit_b --bbox=[65,250,631,940] --task run --savefile
  ```

  - retire le fond et sauvegarde les résultats mask+img_without_bg (noms par défaut)
  ```bash
  python ia_sam.py --input=dog.jpeg --model-type=vit_b --bbox=[65,250,631,940] --task background --savefile
  ```

#### **Usage Python :**

- ***CAS 1 : Mêmes arguments qu'en ligne de commande :***
```python
import ia_sam

args = {}
args["task"] = ...
agent = ia_sam.run_process(args)
agent.results
  # agent.results["img_without_bg"]
  # agent.results["mask"]
```

- ***CAS 2 : Utilisation manuel et plus fine :***
```python
from ia_sam import Agent_SAM

args = {}
args["task"] = "background"
args["model_type"] = "vit_b"
args["input"] = "dogs.jpeg"
args["bbox"] = "[65,250,631,940]"
image_rgb = np.array(Image.open(args["input"]).convert("RGB"))
args["sam_img_in"] = image_rgb
agent = Agent_SAM(model_type=args["model_type"])
agent.run(args, mode=args["task"])
agent.save_results()
```

- Exemples :
  - retire le fond puis fait quelques choses avec les résultats (CAS 1)
  ```python
  from ia_sam import Agent_SAM

  args = {}
  args["model_type"] = "vit_b"
  args["task"] = "background"
  args["input"] = "dog.jpeg"
  args["bbox"] = "[65,250,631,940]"
  agent = ia_sam.run_process(args)
  agent.results
    # agent.results["img_without_bg"]
    # agent.results["mask"]
  ```
  - et sauvegarde (CAS 2)
  ```python
  import numpy as np
  from PIL import Image
  from ia_sam import Agent_SAM

  # 1- Préparation des arguments
  args = {}
  args["task"] = "background"
  args["model_type"] = "vit_b"
  args["input"] = "dogs.jpeg"
  args["bbox"] = "[65,250,631,940]"
  image_rgb = np.array(Image.open(args["input"]).convert("RGB"))
  args["sam_img_in"] = image_rgb
  
  # 2- Exécution
  agent = Agent_SAM(model_type=args["model_type"])
  agent.run(args, mode=args["task"])
  agent.save_results()
  ```

### Agent LISA (`Agent_LISA`)

L'agent LISA permet d'englober le fonctionnement de LISA, le LLM, afin de réaliser différentes tâches nécessaire tout le long du pipeline.

**Agent LISA (pour plus de détail voir '§usage' :**
- Entrée :
  - des prompts (utilisateurs, expert, ...)
- Réalisation de la tâche :
  - Tâche LLM (VQA) : répondre à une question
- Sortie (selon la tâche voulue) :
  - images
  - prompt expert à destination d'un agent
  - prompt final utilisateur
  - résultats ou consignes supplémentaires


#### **Installation :**
```bash
conda create --name myenv_lisa python=3.10
conda activate myenv_lisa
python -m pip install --upgrade pip wheel setuptools
cd experiments/
git clone https://github.com/dvlab-research/LISA.git
python -m pip install -r LISA/requirements.txt
python -m pip install flash-attn==2.0.9 --no-build-isolation
conda deactivate
conda deactivate
```
Pour LISA++ :
```
conda activate myenv_lisa
python -m pip install scikit-learn
cd experiments/LISA/
git switch lisa_plus
conda deactivate
```

#### **Usage en ligne de commande :**
```
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
```

- **Exemples :**
  - Exécution de LISA avec une question en sauvegardant le résultat
  ```bash
  CUDA_VISIBLE_DEVICES=1 python ia_lisa.py --input_img=LISA/1466_2L1_cut.jpg  --input_prompt "There is a defect ? Explain in details." --device cuda --savefile
  ```

  - Exécution de LISA avec le prompt utilisateur et les prompts des experts (agent IA) dans un fichier et sauvegarde la réponse
  ```bash
  CUDA_VISIBLE_DEVICES=1 python ia_lisa.py --input_img=../data/MMAD/MVTec-AD/carpet/test/cut/000.png --input_prompt ./user_prompt.txt --input_expert ./expert_prompt.txt --device cuda --savefile
  ```

#### **Usage Python :**

- ***CAS 1 : Mêmes arguments qu'en ligne de commande :***
```python
import ia_lisa

args = {}
args["task"] = ...
agent = ia_lisa.run_process(args)
agent.results
  # agent.results["text_output"]
  # agent.results["pred_masks"]
```

- ***CAS 2 : Utilisation manuel et plus fine :***
```python
from ia_lisa import Agent_LISA

args = {}
args["task"] = ...
agent = Agent_LISA(args)
agent.run(args)
agent.save_results()
```

- Exemples :
  - CAS 2
  ```python
  import ia_lisa
  from ia_lisa import Agent_SAM

  # 1- Préparation des arguments
  args = {}
  args["input_prompt_str"] = "There is an defect ? explain in details"
  args["input_expert_str"] = ""
  args["version"] = ia_lisa.DEFAULT_LISA_MODEL
  args["precision"] = "bf16"
  args["image_size"] = 1024
  args["load_in_8bit"] = False
  args["load_in_4bit"] = False
  args["model_max_length"] = ia_lisa.DEFAULT_LISA_MODEL_MAX_LENGTH
  args["lora_r"] = ia_lisa.DEFAULT_LISA_MODEL_LORA
  args["vision_tower"] = ia_lisa.DEFAULT_LISA_MODEL_VISION_TOWER
  args["use_mm_start_end"] = ia_lisa.DEFAULT_LISA_MODEL_USE_MM_START_END
  args["conv_type"] = ia_lisa.DEFAULT_LISA_MODEL_CONV_TYPE
  args["device"] = ia_lisa.DEVICE_GLOBAL
  image_rgb = np.array(Image.open(args["input_img"]).convert("RGB"))
  args["sam_img_in"] = image_rgb
  #local_arg["lisa_img_in"] = agent_sam.results["img_without_bg"]

  # 2- Exécution
  agent_lisa = Agent_LISA(args)
  agent_lisa.run(args)
  agent.save_results()
  ```
