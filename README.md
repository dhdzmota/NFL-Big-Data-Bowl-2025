# NFL Big Data Bowl 2025


Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── extra          <- Data from third party sources.
    │   ├── processed      <- Intermediate data that has been transformed.
    │   ├── final          <- The final, canonical data sets for modeling.
    │   └── original       <- The original, immutable data dump. 
    │        │                This original dataset contains a handcrafted dataset 
    │        │                named position_dictionary.csv and a folder that has 
    │        │                all the tracking data.
    │        └── tracking  <- Folder that contains files of tracking_week (1-9). 
    │
    ├── documents          <- Folder with documents relevant to the planning of 
    │                         the proyect. It contains another folder called 
    │                         `images` which contains a logo for some plots.
    │
    ├── figures             <- Folder where other folders reside with the name
    │                          refering to the gameId_playId containing all the
    │                          snapshots of the field and model evaluations. 
    │
    ├── models             <- Folder where model is saved along with the 
    │                         requiered generated sub models such as encoders 
    │                         or non-supervised models that were used to create
    │                         features. Each model is saved in a different
    │                         folder; the `final_model` contains the last 
    │                         generated model.
    │
    ├── notebooks          <- Jupyter notebooks that helped to develop the proyect. 
    │
    ├── outputs            <- Contains GIF files of evaluation of the model.
    │
    │
    ├── reports            <- Contains main analysis for the contest: `final_report.ipynb`
    │   └── report-imgs    <- Generated graphics and figures to be used.
    │
    ├── src                <- Contains all necessarry code to successfully run the repo. 
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`


Steps
-------------
Steps to succesfully execute the code and have your own model at hand:
- Step 0: Create a virtual environment, and execute 
`pip install -r requierements.txt` command.
- Step 1: Manually download the data from the competition and github (if 
cloned, the github data should be already at `data/original` path). The 
original folder should contain `games.csv`, `player_play.csv`, `plays.csv`, 
`players.csv`, a handcrafted dataset `position_dictionary.csv` and an 
additional folder inside `tracking`. This second folder should contain files 
with the names from tracking_week_1.csv to tracking_week_9.csv. This is the 
only information required to train the model.
- Step 2: In root location, execute the following command: 
`python general_pipeline.py`, this will automatically execute the corresponding
scripts to create features and train the model locally which will be saved at 
`models` file. 
- Step 3: To see how to use the model, you can quickly use the file on `src`
named `model_use.py`, which will yield from a specific gameId and playId, the
corresponding model predictions.
