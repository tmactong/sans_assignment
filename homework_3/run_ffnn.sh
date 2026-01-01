#!/usr/bin/env bash
set -e

TRAIN_SIZE=0.1
EPOCHS=500
BATCH=128

HIDDENS=(
  "32" "64" "128" "256"
  "64,32" "128,64" "256,128"
  "128,64,32" "256,128,64"
)

LRS=("1e-4" "3e-4" "1e-3" "3e-3" "1e-2")

echo "train_size,hidden,lr,train_r2,train_rmse,test_r2,test_rmse,overall_r2,overall_rmse" > ffnn_sweep.csv

for h in "${HIDDENS[@]}"; do
  for lr in "${LRS[@]}"; do
    echo "Running hidden=${h}, lr=${lr}"
    out=$(python ffnn.py --train-size ${TRAIN_SIZE} --hidden "${h}" --lr ${lr} --batch-size ${BATCH} --max-epochs ${EPOCHS})

    # Extract the markdown table row like: |ffnn|train_r2|train_rmse|test_r2|test_rmse|overall_r2|overall_rmse|
    row=$(echo "$out" | grep -E '^\|ffnn\|' | tail -n 1)

    # Parse by '|' delimiter. Fields: 1 empty, 2 model, 3..8 metrics, 9 empty
    metrics=$(echo "$row" | awk -F'|' '{print $3","$4","$5","$6","$7","$8}')

    echo "${TRAIN_SIZE}\t${h}\t${lr}\t${metrics}" >> ffnn_sweep.csv
  done
done

echo "Saved to ffnn_sweep.csv"