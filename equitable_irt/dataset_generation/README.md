This subpackage processes the raw camera and CSV data.

Make sure the data follows this directory structure (the landmarks folder is created after running the landmark code).

```
dataset_dir
|-- data.csv
|-- data
    |-- name
        |-- base
            |-- ir0.png
            |-- rgb0.png
        |-- cool
            |-- ir0.png
            |-- rgb0.png
|-- landmarks
    |-- name
        |-- base.json
        |-- cool.json
```

Example code is in `prepare_dataset.ipynb`. The code does the following:
1. Generate and save landmarks
2. Create instances of Subject classes
    * Aligns and crops data
    * Generates surface normals
3. Save ML dataset
    * Saves cropped images
    * Saves CSV with labels and bounding boxes
