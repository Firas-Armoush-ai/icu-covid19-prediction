# Raw data

`Kaggle_Sirio_Libanes_ICU_Prediction.xlsx` is the source file from the
Hospital Sírio-Libanês **COVID-19 – Clinical Data to assess diagnosis** dataset
(Kaggle, version 4).

* Source: https://www.kaggle.com/datasets/S%C3%ADrio-Libanes/covid19
* Licence: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Non-commercial use only, with attribution. The repository's MIT License covers the code, not this data.
* Citation: see [Dataset source](../../README.md#dataset-source) in the main README.

## What the file contains

* 385 patients × 5 time windows = 1,925 rows, 231 columns.
* Windows: 0–2 h, 2–4 h, 4–6 h, 6–12 h and above 12 h from admission.
* Period covered: 28 January – 16 April 2020; Hospital Sírio-Libanês sites in São Paulo and Brasília.
* The provider anonymised the data and min-max scaled every numeric column to the range −1 to 1 before publication. A model trained on it therefore expects these scaled values, not raw clinical units.

## Processed data

`data/processed/df_model.csv` is derived from this file. The steps are described in the
[Licence](../../README.md#licence) section of the main README.
