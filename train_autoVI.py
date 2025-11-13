from anomalib.data import Folder
from anomalib.models.image import EfficientAd
from anomalib.models.image import Patchcore
from anomalib.models.image import Padim
from anomalib.engine import Engine

def main():
    # 1. Charger AutoVI v1 (ex: engine_wiring)
    data = Folder(
        name="engine_wiring",                          # nom de la classe
        root="datasets/AutoVI_v1/engine_wiring",              # chemin vers le dataset décompressé
        normal_dir="train",                            # images normales
        abnormal_dir="test",                           # images anormales
        #mask_dir="ground_truth",                       # masques des anomalies
        extensions=[".png", ".jpg"],                   # formats acceptés
        train_batch_size=1,   # Taille du lot d’images pendant l’entraînement.
                              # Pour EfficientAD, c’est obligatoirement 1 (spécificités du modèle).
        eval_batch_size=1     # Taille du lot pendant validation/test.
                              # Pour EfficientAD, on garde 1 aussi pour rester aligné (et éviter les surprises mémoire/comportement).
                              # Pour PatchCore/PaDiM, on peut monter (ex. 16, 32) si la RAM/VRAM le permet.
    )

    # 2. Choisir le modèle
    model = EfficientAd()

    # 3. Créer le moteur d'entraînement
    engine = Engine(
        max_epochs=1,              # Nombre de passes complètes sur le dataset d’entraînement.
                                   # 1 = juste un smoke test rapide (ce qu'on a fait).
                                   # Pour un benchmark fiable, on peut viser 20+ (selon le modèle/dataset).
        default_root_dir="results" # dossier où seront sauvegardés les résultats
    )

    # 4. Lancer l'entraînement + test
    engine.fit(model=model, datamodule=data)
    engine.test(model=model, datamodule=data)

if __name__ == "__main__":
    main()
