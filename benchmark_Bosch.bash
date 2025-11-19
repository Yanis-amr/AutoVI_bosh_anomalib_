#!/bin/bash

experiment_name="2025_BOSCH_bench"
classes_names=("BOSCH")                        # une seule classe
models_names=("Padim" "Patchcore" "EfficientAd")

for class_name in "${classes_names[@]}"; do
  for model_name in "${models_names[@]}"; do
    echo "Running test.py for class: $class_name & model: $model_name"
    python3 test.py --experiment_name $experiment_name \
                    --class_name $class_name \
                    --model_name $model_name
    echo "---------------------------"
  done
done
