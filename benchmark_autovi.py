# benchmark_autovi.py

# Mini-benchmark sur AutoVI v1 / engine_wiring en entraînant successivement EfficientAD, PatchCore et PaDiM (1 époque, CPU), 
# puis affiche un tableau récapitulatif des scores (AUROC/F1 image, et pixel si dispo).

# --- PATCH ANOMALIB pour Windows: désactiver totalement les liens symboliques : symlinks "latest" ---
from pathlib import Path
from anomalib.utils import path as an_path

def _no_symlink_versioning(root_dir: Path) -> Path:
    """Crée simplement le dossier sans 'latest' ni lien symbolique."""
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir

an_path.create_versioned_dir = _no_symlink_versioning
# -------------------------------------------------------------------------------

from anomalib.data import Folder
from anomalib.models.image import EfficientAd, Patchcore, Padim
from anomalib.engine import Engine
from typing import Dict, Any, List

# --- Réglages de base ---
DATA_ROOT = "datasets/AutoVI_v1"        # racine de ton dataset AutoVI v1
CATEGORY  = "engine_wiring"             # sous-dossier à tester
EXTS      = [".bmp", ".png", ".jpg"]
RESULTS_DIR = "results"
MAX_EPOCHS = 1                          # On reste à 1 epoch pour que ca tourne sur un portable : 1 passage complet du modèle sur toutes les images d’entraînement

# Pour éviter les soucis CPU/Windows (démarrage plus simple des DataLoaders)
COMMON_DATA_KW = dict(
    num_workers=0,  # évite les overheads de workers sur CPU Windows
)

# Configs par modèle (nom lisible, constructeur, batch train, batch eval)
MODEL_CONFIGS = [
    ("EfficientAd", EfficientAd, 1, 1),  # EfficientAd exige batch_size=1 -> le modèle traite une image à la fois.
    ("PatchCore",  Patchcore,    8, 8),  # PatchCore supporte des batchs >1 -> il traite 8 images en parallèle, plus rapide mais plus gourmand en RAM.
    ("PaDiM",      Padim,        8, 8),  # PaDiM aussi
]

# Fonction qui éxecute le benchmark pour un modèle donné
def run_one(model_name, model_ctor, train_bs, eval_bs):
    # 1) data
    data = Folder(
        name=CATEGORY,
        root=f"{DATA_ROOT}/{CATEGORY}",
        normal_dir="train",
        abnormal_dir="test",
        #mask_dir="ground_truth", #on peut activer les masques d'anomalie si on veut des métriques au niveau pixel
        extensions=EXTS,
        train_batch_size=train_bs,
        eval_batch_size=eval_bs,
        **COMMON_DATA_KW,
    )

    # 2) model
    model = model_ctor()

    # 3) dossier résultats (sans symlinks grâce au patch)
    results_dir = Path(RESULTS_DIR) / model_name / CATEGORY

    # 4) engine (aucun argument exotique)
    engine = Engine(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        logger=False,
        default_root_dir=results_dir,
    )

    # Sur la tour de bureau, on peut activer le GPU et checkpointing :
    #engine = Engine(
    #max_epochs=20,
    #accelerator="gpu",
    #devices=1,
    #logger=True,
    #enable_checkpointing=True,
    #default_root_dir=results_dir,
    #)

    # 5) train + test
    engine.fit(model=model, datamodule=data)
    engine.test(model=model, datamodule=data)

    # 6) récupération métriques
    metrics = {}
    try:
        for k, v in engine.trainer.callback_metrics.items():
            try:
                metrics[k] = float(v)
            except Exception:
                pass
    except Exception:
        pass
    return metrics

def main():
    rows: List[Dict[str, Any]] = []
    print(f"\n=== AutoVI v1 benchmark (max_epochs={MAX_EPOCHS}, CPU) ===\n")
    for name, ctor, train_bs, eval_bs in MODEL_CONFIGS:
        print(f"-> Running {name} (train_bs={train_bs}, eval_bs={eval_bs}) ...")
        m = run_one(name, ctor, train_bs, eval_bs)
        if not m:
            print(f"   [WARN] Aucune métrique récupérée pour {name}")
            rows.append({"model": name})
        else:
            keys_sorted = sorted(m.keys())
            pretty = ", ".join(f"{k}={m[k]:.4f}" for k in keys_sorted)
            print(f"   Metrics: {pretty}")
            row = {"model": name}
            row.update({k: m[k] for k in keys_sorted})
            rows.append(row)
        print()

    # Tableau final
    if rows:
        all_keys = {"model"}
        for r in rows:
            all_keys.update(r.keys())
        all_keys = [k for k in ["model","image_AUROC","image_F1Score","pixel_AUROC","pixel_F1Score"] if k in all_keys] + \
                   [k for k in all_keys if k not in {"model","image_AUROC","image_F1Score","pixel_AUROC","pixel_F1Score"}]

        col_widths = {k: max(len(k), max(len(f"{r.get(k,'')}") for r in rows)) for k in all_keys}
        sep = " | "

        print("=== Résumé ===")
        header = sep.join(k.ljust(col_widths[k]) for k in all_keys)
        print(header)
        print("-" * len(header))
        for r in rows:
            line = sep.join(
                f"{r.get(k,'')}".ljust(col_widths[k]) if not isinstance(r.get(k,""), float)
                else f"{r.get(k):.4f}".ljust(col_widths[k])
                for k in all_keys
            )
            print(line)

if __name__ == "__main__":
    main()
