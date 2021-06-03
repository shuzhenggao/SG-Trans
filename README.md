Our code is adapted from https://github.com/wasiahmad/NeuralCodeSum
### Training Models

You can train our model with following steps.

```
$ cd  DATASET_NAME/scripts/DATASET_NAME
```

Where, choices for DATASET_NAME are ["java", "python"].

To train a model, run:

```
$ bash script_name.sh GPU_ID MODEL_NAME
```

However, due to the restriction of the size of uploaded data, we don't upload our parsed data and the file  'meteor-1.5.jar' to calculate METEOR score.
