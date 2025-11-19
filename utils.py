from pathlib import Path
from anomalib.utils import path as an_path

def _no_symlink_versioning(root_dir: Path) -> Path:
    """Crée simplement le dossier sans 'latest' ni lien symbolique (patch Windows)."""
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir

# On remplace la fonction d’origine par notre version sans symlinks
an_path.create_versioned_dir = _no_symlink_versioning


from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim, Draem, Dsr, Cflow, EfficientAd
from anomalib.metrics import AUROC, AUPR, AUPRO
from anomalib.data import Folder
from anomalib.deploy import ExportType
from anomalib.deploy import TorchInferencer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, average_precision_score, roc_auc_score
from itertools import chain
import pandas as pd
import time
import matplotlib.pyplot as plt
import torch
torch.set_float32_matmul_precision('medium')
import argparse
from PIL import Image
#MAX_EPOCHS = 1

if model_name == "EfficientAd":
    MAX_EPOCHS = 30  # voire 50 sur une grosse tour
else:
    MAX_EPOCHS = 1

if model_name == "EfficientAd":
    train_bs = eval_bs = 1
elif model_name in ["Patchcore", "Padim"]:
    train_bs = eval_bs = 32   # ou 64 si GPU 16 Go


# version pour AutoVI sans masques d’anomalie
""" def init_datamodule(class_name):
    datamodule = Folder(
        name=class_name,
        root="datasets/AutoVI_v1/" + class_name,
        normal_dir="train/good",
        abnormal_dir="test",          # test = toutes les images pour l’éval
        # mask_dir="ground_truth",    # ❌ on COMENTE / SUPPRIME cette ligne
        train_batch_size=5,
        eval_batch_size=5,
        num_workers=0,                # sous Windows, plus sûr
        extensions=[".bmp", ".png", ".jpg"],
    )
    datamodule.setup()
    return datamodule """


# version pour BOSCH sans masques d’anomalie
def init_datamodule(class_name, task="classification"):
    #Datamodule pour BOSCH avec split manuel train/test.

    datamodule = Folder(
        name=class_name,
        root="./datasets/" + class_name,   # ./datasets/BOSCH
        normal_dir="train/normal",         # train = only OK
        abnormal_dir="test/abnormal",      # test/abnormal = NOK
        mask_dir=None,                     # pas de masques
        extensions=[".bmp", ".png", ".jpg"],
        train_batch_size=train_bs,
        eval_batch_size=eval_bs,
        # ⚠️ pas de test_split_ratio ici, tu as déjà split à la main
        # pas de val_split_mode : anomalib construira train/test,
        # la validation risque d'être vide ou dérivée du test selon la version.
        num_workers=8,
    )

    datamodule.setup()

    print(
        "Datamodule:", datamodule.name, "\n",
        "   train data:", np.sum(datamodule.train_data.samples.label_index == 0),
        "images OK ;", np.sum(datamodule.train_data.samples.label_index == 1),
        "images NOK", "\n",
        "   test data:", np.sum(datamodule.test_data.samples.label_index == 0),
        "images OK ;", np.sum(datamodule.test_data.samples.label_index == 1),
        "images NOK",
    )

    return datamodule




def train(datamodule, model, output_dir = "./results/"):
    ## Init anomalib Engine
    #engine = Engine(image_metrics=None, pixel_metrics=None, default_root_dir=output_dir) #ancienne version d'anomalib
    engine = Engine(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu", # sur pc tour sinon cpu
        devices=1, # ne pas écrire cela sur pc portable         
        logger=False,              # pas de logger pour simplifier
        default_root_dir=output_dir,
    )

    ## Fit on datamodule train set
    engine.fit(datamodule = datamodule, model=model) 
    pt_path = engine.export(model, ExportType.TORCH)
    return engine

def predict(datamodule, engine):
    ## Return inference results of the trained model on test set
    predictions = engine.predict(dataloaders=datamodule.test_dataloader(), model=engine.model,
                                 ckpt_path=engine.trainer.checkpoint_callback.best_model_path)
    return predictions

def compute_metrics(predictions, output_dir):
    """
    Calcule les métriques image (AUROC, AP, accuracy, tpr@fpr)
    + courbe ROC, en utilisant scikit-learn.

    Pour l’instant, les métriques pixel (Pixel_Auroc, Pixel_Aupro)
    sont mises à 0 (pas de masques exploités).
    """

    # On empile tous les labels / scores / prédictions
    y_true = np.array(list(chain(*[np.array(p["label"]).ravel() for p in predictions])))
    y_scores = np.array(list(chain(*[np.array(p["pred_scores"]).ravel() for p in predictions])))
    y_pred = np.array(list(chain(*[np.array(p["pred_labels"]).ravel() for p in predictions]))).astype(int)

    # --- métriques image (classification) ---
    # AUROC "classique"
    image_auroc = roc_auc_score(y_true, y_scores)

    # AP (Average Precision)
    ap = average_precision_score(y_true, y_scores)

    # Accuracy à partir des labels prédits
    accuracy = accuracy_score(y_true, y_pred)

    # Courbe ROC pour :
    #  - sauvegarder un plot
    #  - calculer tpr @ fpr = [0, 0.01, 0.05, 0.1]
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    tpr_values = np.interp([0, 0.01, 0.05, 0.1], fpr, tpr)
    image_auroc2 = auc(fpr, tpr)

    # --- Sauvegarde de la courbe ROC ---
    plt.clf()
    plt.title("ROC")
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % image_auroc2)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(str(Path(output_dir) / "auroc.png"))

    # --- métriques pixel : pour l'instant, 0 ---
    pixel_auroc = 0.0
    pixel_aupro = 0.0

    metrics = pd.DataFrame(
        {
            "Image_Auroc": [np.around(image_auroc, 3)],
            "AP": [np.around(ap, 3)],
            "tpr@fpr=n": [np.around(tpr_values, 3)],
            "accuracy": [np.around(accuracy, 3)],
            "Pixel_Auroc": [np.around(pixel_auroc, 3)],
            "Pixel_Aupro": [np.around(pixel_aupro, 3)],
        }
    )
    return metrics


def cls_result(predictions):
    images_paths =np.array(list(chain(*[pred['image_path'] for pred in predictions])))
    images_names = [path.split("\\")[-1] for path in images_paths] 
    y_true = np.array(list(chain(*[pred['label'] for pred in predictions])))
    y_scores = np.array(list(chain(*[pred['pred_scores'] for pred in predictions])))
    y_pred = np.array(list(chain(*[pred['pred_labels'] for pred in predictions]))).astype(int) 

    result = pd.DataFrame(
        {'image_name': images_names,
        'y_true': y_true,
        'y_score': y_scores,
        'y_pred': y_pred,
        }
    )
    return result

def inference_torch(image_path, torch_model, show_result=False):
    inferencer = TorchInferencer(path=torch_model)
    prediction = inferencer.predict(image = image_path)
    print('predicted label:', prediction.pred_label, '  ', 'predicted score:', prediction.pred_score)
    #prediction.heat_map

    if True:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(prediction.image)
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        ax[1].imshow(prediction.heat_map)
        ax[1].set_title("predict Heat Map")
        ax[1].axis('off')
        ax[2].imshow(prediction.pred_mask)
        ax[2].set_title("Predict Mask")
        ax[2].axis('off')
        title = 'predict label:  ' + str(prediction.pred_label.value) + '       ' + 'predict score:  ' + str(np.around(prediction.pred_score, 4))
        color = 'red' if prediction.pred_label.value == 1 else 'green'
        #fig.suptitle(title, fontsize=10, fontweight = 2, color=color)
        fig.text(0.3, 0.1, title, fontsize=20, fontweight=1000, color = color)
        fig.patch.set_linewidth(20)
        fig.patch.set_edgecolor(color)
        fig.savefig(result_image)
        plt.close(fig)
        if show_result:
            img = Image.open(result_image)
            img.show()



    return prediction

def experiment_metrics_synthesis(experiment_name):
    aurocs_table = pd.DataFrame()
    aps_table = pd.DataFrame()
    aupros_table = pd.DataFrame()

    experiment_name = "202502_AutoVI_remaster_bench"
    methods = os.listdir("./results/" + experiment_name)

    for method in methods:
        aurocs_, aps_, aupros_ =[],[],[]   
        classes = os.listdir("./results/" + experiment_name + "/" + method)
        for class_name in classes:
            results = pd.read_excel("./results/" + experiment_name + "/" + method + "/" + class_name + "/metrics.xlsx", index_col=0)

            aurocs_.append([np.around(np.mean(results.Image_Auroc)*100, 1), np.around(np.std(results.Image_Auroc)*100, 1)])
            aps_.append([np.around(np.mean(results.AP)*100, 1), np.around(np.std(results.AP)*100, 1)])
            aupros_.append([np.around(np.mean(results.Pixel_Aupro)*100, 1), np.around(np.std(results.Pixel_Aupro)*100, 1)])

        aurocs_.append(np.around(np.mean(aurocs_, axis=0), 1).tolist())
        aurocs_table[method] = aurocs_

        aps_.append(np.around(np.mean(aps_, axis=0), 1).tolist())
        aps_table[method] = aps_

        aupros_.append(np.around(np.mean(aupros_, axis=0), 1).tolist())
        aupros_table[method] = aps_


    classes.append('mean')
    aurocs_table.insert(0, '', classes)
    aps_table.insert(0, '', classes)
    aupros_table.insert(0, '', classes)

    aurocs_table.to_excel("./results/" + experiment_name + "/Auroc_mean.xlsx")
    aps_table.to_excel("./results/" + experiment_name + "/AP_mean.xlsx")
    aupros_table.to_excel("./results/" + experiment_name + "/Aupro_mean.xlsx")



