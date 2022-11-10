# The Program for Evaluating the Accuracy

The usage of the program for evaluating the accuracy(error) which is defined in the competition.

## Requirements

- Python 3
- Libraries
  - numpy

## Usage

Move to the directory in which "evaluate.py" exists, and excecute the following command in your terminal, then the score(error) for each scene and the overall score(error) will be printed.

```bash
python evaluate.py --ground-truth-path /path/to/ground/truth --predictions-path /path/to/predictions --meta-data-path /path/to/meta/data
```

- Specify the path to the answer file and the prediction file to the arguments "--ground-truth-path" and "--predictions-path" respectively.
- The file format of the answer and the prediction is json, and the details should be as follows. The key and the value should be the scene id and the velocities listed in chronological order, respectively.

```json
{
    "000": [...],
    "001": [...],
    ...
}
```

- Specify the path to the parameter file required for the evaluation to the argument "--meta-data-path".
- The format of the parameter file is json, and the details are as follows.

```json
{
    "first_frame": 20,
    "limit_gradient": 0.07,
    "limit_intercept": 3,
    "weights": 
    {
        "000": 1,
        ...
    }
}
```

- The first frame number the evaluation starts is specified in "first_frame"(the frame number starts with 1).
- The parameters to determine the upper limit of the velocity error for each frame are specified in "limit_gradient" and "limit_intercept".
  - "limit_gradient" and "limit_intercept" are the gradient and the y-intercept of the linear function, respectively.
  - Please also refer to the "Evaluation" tab in the competition site.
- The weight for each scene is specified in "weights". The key is the scene id, and the value is the weight.
  - If "評価値計算時の重み付加"="有" 3, 1 otherwise(private in testing).
- The length of the predicted velocities in the prediction file should be the same in all of the scenes as the answer file.
- "weights" in the parameter file should correspond to the answer file and the prediction file.
- The sample files for the answer, the prediction, and the parameter are under "./data/". Please refer to them if needed.