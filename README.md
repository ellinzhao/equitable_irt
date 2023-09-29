## Pipeline
Assume we have data from a subject in the `0110_name` folder. Make sure the metadata for the subject is in the CSV file.
1. Detect and save landmarks:
```
from landmarks_regions import process_dir

data_dir = '0110_name'
save_dir = 'lms/'
lms = process_dir(data_dir, save_dir)
```
2. Load data as an instance of `Subject` class, which will clean the data using the landmarks (remove NUC spikes, crop images to the face):
```
from classes import Subject

dataset_dir = './'  # folder that contains all the data
name = '0110_name'
sub = Subject(dataset_dir, name, units='F')
```

3. Save new image files to be used by the ML model:
```
sub.save_dataset()

# The saved data can be modified in RGB_IR_Data.save_ir()
```

## Misc
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
