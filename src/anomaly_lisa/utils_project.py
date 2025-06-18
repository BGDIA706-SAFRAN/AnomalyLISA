"""Fichier global de fonctions utilitaires.

===========
Fonctions :
    dict_take(n, iterable, make_dict=False) -> list | dict
    dict_revert(x: dict, sort: bool = False) -> list
    g_cleaner(to_del: list[str] | str) -> None

    # Fonction en relation avec Torch en mode leasy import :
    torch_pick_device(gpu_device_name="cuda", test_directx=True) -> torch.device
    torch_check_test(gpu_device_name="cuda", test_directx=True, fct_test=None, N=10000) -> None
    torch_check_info(gpu_device_name="cuda", test_directx=True, do_tests=True, **args) -> None
"""
__author__ = ['Nicolas Allègre']
__date__ = '17/03/2025'
__version__ = '0.5'

###############################################################################
# IMPORTS :
# /* Modules standards */
import gc
import time
from collections.abc import Callable, Iterable
from itertools import islice


###############################################################################
# FONCTIONS GLOBALES :
def dict_take(n: int, iterable: Iterable, make_dict: bool = False) -> list | dict:
    """Return the first n items of an iterable.

    As a list (like dict), or keep in dict for a dict if make_dict.
    """
    result = []
    if make_dict and isinstance(iterable, dict):
        result = dict(list(iterable.items())[:n])
    else:
        result = list(islice(iterable, n))
    # end if
    return result
# end def dict_take


def dict_revert(x: dict, sort: bool = False) -> list:
    """Renverse un dictionnaire key/value en matrice 2 x n (key, value).

    Notamment utile pour afficher via tabulate.
    """
    tmp = x
    if sort is True:
        tmp = dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
    # end if
    return [[list(tmp.keys())[i], list(tmp.values())[i]] for i in range(len(tmp))]
# end def dict_revert


def g_cleaner(to_del: list[str] | str) -> None:
    """Global safe variable memory cleaner (CPU&GPU).

    Exemple:
        to_del = "models, model, my_resnet, resnet, dataset_cifar"
        g_cleaner(to_del)
    """
    if isinstance(to_del, str):
        to_del = to_del.replace(" ", "").split(",")
    # end if
    for var_id in to_del:
        globals().pop(var_id, None)
    # end for
    _ = gc.collect()
    if "torch" in globals():
        globals()["torch"].cuda.empty_cache()
# end def g_cleaner


def torch_pick_device(gpu_device_name: str = "cuda", test_directx: bool = True):
    """Choisi un device pour torch, en prennant le meilleur (GPU sinon CPU).

    Torch < 2.6 :
    # device = torch.device(gpu_device_name if torch.cuda.is_available() else "cpu")
    Torch >= 2.6 :
    # device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
    torch_directml
    # device = torch.device("cuda" if torch.cuda.is_available() else torch_directml.device() if torch_directml.device() else "cpu")

    :param gpu_device_name:  choix manuel du device GPU (cuda/mps/xpu/vulkan) pour Torch <2.6 (auto en >=2.6)
        'cuda si Nvidia, 'mps' si Mac, 'xpu' si GPU Intel
    :param test_directx:  check DirectX via torch_directml si non CUDA
    :return torch.device: le device le plus approprié permettant l'accélaration matériel
    """
    # 1- Import torch local pour éviter l'import des codes sans besoin de Torch.
    import torch

    # 2- CPU par défaut
    device = torch.device("cpu")
    torch_version = int("".join(torch.__version__.split(".")[:2]))
    # 3- Choix GPU supporté par défaut par Torch (Nvida, MAC/MPS, INTEL/XPU)
    if torch.cuda.is_available():  # Il y a un GPU disponible
        device = torch.device(gpu_device_name)
        # 3.1 - si torch>=2.6, gestion automatique cuda/mps/xpu via accelerator
        if torch_version >= 25 and torch.accelerator.is_available():
            device = torch.device(torch.accelerator.current_accelerator().type)
        # end if
    # 4- Choix GPU tier avec DirectX
    elif test_directx:
        import torch_directml
        if torch_directml.device():
            device = torch.device(torch_directml.device())
    # end if
    return device
# end def torch_pick_device


def torch_check_test(gpu_device_name: str = "cuda", test_directx: bool = True,
                     fct_test: Callable | None = None, N: int = 10000) -> None:
    """Effectue un test de performance sur tout les devices de la machine.

    :param gpu_device_name:  choix manuel du device GPU
    :param test_directx:  check avec DirectX via torch_directml
    :param fct_test:    fonction pour effectuer des calculs complexe pour les tests.
        (Callable[[torch.Tensor, torch.Tensor], str])
    :param N:   taille des tensor # Adjust this size based on your system's capabilities
    """
    import torch

    # 1- Création des devices à tester
    devices_to_check = []
    results = []
    # CPU
    devices_to_check.append(torch.device("cpu"))
    results.append({"device": "cpu"})
    # CUDA ou autre
    if torch.cuda.is_available():
        devices_to_check.append(torch.device(gpu_device_name))
        results.append({"device": gpu_device_name})
        # MAC/MPS
        if torch.backends.mps.is_available() and gpu_device_name != "mps":
            devices_to_check.append(torch.device("mps"))
            results.append({"device": "mps"})
        # INTEL/XPU
        if torch.xpu.is_available() and gpu_device_name != "xpu":
            devices_to_check.append(torch.device("xpu"))
            results.append({"device": "xpu"})
    # DIRECTX
    if test_directx:
        import torch_directml
        if torch_directml.device():
            devices_to_check.append(torch.device(torch_directml.device()))
            results.append({"device": "DirectML"})

    # 2- Définition du test
    def complex_operations(tensor1, tensor2):
        # Perform a series of intensive operations
        result_mm = torch.mm(tensor1, tensor2)
        result_elementwise = tensor1 * tensor2
        result_sum = torch.sum(result_elementwise)
        return result_mm, result_elementwise, result_sum

    # 3- Exécution des tests
    # Create random tensors
    tensor1 = torch.rand(N, N, device="cpu")
    tensor2 = torch.rand(N, N, device="cpu")
    for i, device in enumerate(devices_to_check):
        result = results[i]
        print(device.type, "testing ...", end='', flush=True)
        # Create random tensors
        tensor1 = tensor1.to(device)
        tensor2 = tensor2.to(device)
        # Execute tests
        start_time = time.time()
        result_mm, result_elementwise, result_sum = complex_operations(tensor1, tensor2)
        if fct_test is not None:
            result_additionnel = fct_test(tensor1, tensor2)
        end_time = time.time() - start_time
        # Résultats
        # To avoid overwhelming the output, only show the shape of the results
        result["shape_mm"] = result_mm.shape
        result["shape_elementwise"] = result_elementwise.shape
        result["sum"] = result_sum.item()
        result["time"] = f"{end_time:.4f} seconds"
        if fct_test is not None:
            result["additionnel"] = result_additionnel
        results.append(result)
        print(" => ", end_time)

    # 4- Afficher les résultats
    for i, device in enumerate(devices_to_check):
        result = results[i]
        print(f"** {result["device"]} : **")
        print(f"\t Time taken : {result["time"]}")
        print(f"\t Result Sum: {result["sum"]:.4f}")
        print(f"\t Result MM Shape: {result["shape_mm"]}")
        print(f"\t Result Elementwise Shape: {result["shape_elementwise"]}")
        if fct_test is not None:
            print(f"\t Result additionnel: {result["additionnel"]}")
# end def torch_check_test


def torch_check_info(gpu_device_name: str = "cuda", test_directx: bool = True,
                     do_tests: bool = True, **args) -> None:
    """Affiche les informations sur l'accélaration matériel.

    :param (str) gpu_device_name:   choix manuel du device GPU
    :param (bool) test_directx:     check avec DirectX via torch_directml
    :param (bool) do_tests:         exécute les tests de performance
    :param **args:      paramètres supplémentaire pour les tests (voir `torch_check_test`)
        N  taille des tensor pour les tests
    """
    # 1- Import torch local pour éviter l'import des codes sans besoin de Torch.
    import torch

    # 2- Info sur Torch et CUDA
    print("** Torch available :", torch.cuda.is_available())
    print("\t device count : ", torch.cuda.device_count())
    print(f"\t {torch.backends.cudnn.version()=}")
    if torch.cuda.is_available():
        print(f"\t {torch.version.cuda=}")
        print("Informations :")
        for i in range(torch.cuda.device_count()):
            print("\t", torch.cuda.get_device_properties(i).name)
        # end for
    if torch.backends.cuda.is_built():
        print(f"{torch.cuda.current_device()=}")
    # end if

    # 3- Info sur DirectX
    if test_directx:
        import torch_directml
        print("** Torch DirectX available :", torch_directml.is_available())
        print(f"\t {torch_directml.device()=}")
    else:
        print("Torch DirectX available : non testé")
    # end if

    # 4- Info sur MPS de MAC
    print("** Torch MAC/MPS available : ", torch.backends.mps.is_available())
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("\t MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("\t MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
        # end if
    # end if

    # 5- Exécution des tests
    if do_tests:
        torch_check_test(gpu_device_name, test_directx, **args)
    # end if
    print("Device pris par défaut : ", torch_pick_device(gpu_device_name, test_directx))
# def torch_check_info
