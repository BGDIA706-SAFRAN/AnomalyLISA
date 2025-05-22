# AnomalyLISA
AnomalyLISA : boosting a VLLM with LISA method and expert agents for Industrial Anomaly Detection


## Type d'agent IA

L'architecture est constitué de plusieurs agent qui sont des IA experts dans leurs tâches orchestré par un pipeline.

### Agent SAM

- **Installation :**
```bash
git clone https://github.com/facebookresearch/segment-anything.git SAM
sed -i \"s/torch.load(f)/torch.load(f, weights_only=True)/g\" SAM/segment_anything/build_sam.py
```

- **Usage :**
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


### Agent LISA

- **Installation :**
```bash
cd experiments/
git clone https://github.com/dvlab-research/LISA.git
cd
conda create --name myenv_lisa python=3.10
conda activate myenv_lisa
python -m pip install --upgrade pip wheel setuptools
cd experiments/LISA/
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.0.9 --no-build-isolation
conda deactivate
conda deactivate
cd
```
Pour LISA++ :
```
conda activate myenv_lisa
python -m pip install scikit-learn
cd experiments/LISA/
git switch lisa_plus
```

- **Usage :**
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
```bash
CUDA_VISIBLE_DEVICES=1 python ia_lisa.py --input_img=LISA/1466_2L1_cut.jpg  --input_prompt "There is a defect ? Explain in details." --device cuda --savefile
CUDA_VISIBLE_DEVICES=1 python ia_lisa.py --input_img=../data/MMAD/MVTec-AD/carpet/test/cut/000.png --input_prompt ./user_prompt.txt --device cuda --savefile
```
