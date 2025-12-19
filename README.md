Run `pip install -e .` while in this directory.

# Dataset generation

The `prepare_dataset.ipynb` notebook is used to generate the data for ML model training.

- We assume that all CSV data in °F.
- We assume that all image data is captured at fps=4.

## Directory structure
```
|-- dataset_dir
    |-- data.csv
    |-- data
        |-- subject
            |-- base
                |-- ir0.png
                |-- rgb0.jpg
            |-- cool
                |-- ir0.png
                |-- rgb0.jpg
    |-- landmarks
        |-- ref.json
        |-- subject
            |-- base.json
            |-- cool.json
    |-- ml_data
        |-- subject
            |-- base.csv
            |-- cool.csv
            |-- base_ir0.png
            |-- cool_ir0.png
```

## `landmark.json` structure
All points are in the IR image coordinates.
```
{
    fname: {
        "rois": {
            "forehead": [[y0, y1], [x0, x1]]
        },
        "landmarks": {
            "chin": [[x0, x1, x2, ...], [y0, y1, y2, ...]]
        }
    }
}
```

# Model training and analysis

Model training and plot generation code is in `ml_training` sub-directory.
