## Misc
Assume that all CSV data in °F.
Assume that all image data is captured at fps=4.


## Directory structure
|-- dir
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


# `landmark.json` structure
All points are in the IR image coordinates.
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
