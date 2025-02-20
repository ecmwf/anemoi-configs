# AIFS-Single-MSE-1.0

## Details

Here, we introduce the **Artificial Intelligence Forecasting System (AIFS)**, a data driven forecast
model developed by the European Centre for Medium-Range Weather Forecasts (ECMWF). 

The operational release of AIFS Single v1.0 marks the first operationally supported AIFS model. Version 1 will
supersede the existing experimental version, [0.2.1 AIFS-single](https://huggingface.co/ecmwf/aifs-single). 
The new version, 1.0, will bring changes to the AIFS single model, including among many others:

- Improved performance for upper-level atmospheric variables (AIFS-single still uses 13 pressure-levels, so this improvement mainly refers to 50 hPa)
- Improved scores for total precipitation.
- Additional output variables, including 100 meter winds, snow-fall, solar-radiation and land variables such as soil-moisture and soil-temperature.

### Description

AIFS is based on a graph neural network (GNN) encoder and decoder, and a sliding window transformer processor,
and is trained on ECMWF’s ERA5 re-analysis and ECMWF’s operational numerical weather prediction (NWP) analyses.

It has a flexible and modular design and supports several levels of parallelism to enable training on
high resolution input data. AIFS forecast skill is assessed by comparing its forecasts to NWP analyses
and direct observational data.

- **Developed by:** ECMWF
- **Model type:** Encoder-processor-decoder model

The full list of input and output fields is shown below:

| Field                                                                                                                                                       | Level type                                                                   | Input/Output |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|--------------|
| Geopotential, horizontal and vertical wind components, specific humidity, temperature                                                                       | Pressure level: 50,100, 150, 200, 250,300, 400, 500, 600,700, 850, 925, 1000 | Both         |
| Surface pressure, mean sea-level pressure, skin temperature, 2 m temperature, 2 m dewpoint temperature, 10 m horizontal wind components, total column water | Surface                                                                      | Both         |
| Soil moisture and soil temperature (layers 1 & 2)                                                                                                           | Surface                                                                      | Both         |
| 100m horizontal wind components, solar radiation (Surface short-wave (solar) radiation downwards and Surface long-wave (thermal) radiation downwards), cloud variables (tcc, hcc, mcc, lcc), runoff and snow fall  | Surface               | Output       |
| Total precipitation, convective precipitation                                                                                                               | Surface                                                                      | Output       |
| Land-sea mask, orography, standard deviation of sub-grid orography, slope of sub-scale orography, insolation, latitude/longitude, time of day/day of year   | Surface                                                                      | Input        |

Input and output states are normalised to unit variance and zero mean for each level. Some of the forcing variables, like orography, are min-max normalised.

#### Model resolution

| | Component | Horizontal Resolution [kms] | Vertical Resolution [levels] |
|---|:---:|:---:|:---:|
| Atmosphere | AIFS-single v1.0 | ~ 36 |  13 |

### Citation

If you use this model in your work, please cite it as follows:

**BibTeX:**

```bibtex
@article{lang2024aifs,
  title={AIFS-ECMWF's data-driven forecasting system},
  author={Lang, Simon and Alexe, Mihai and Chantry, Matthew and Dramsch, Jesper and Pinault, Florian and Raoult, Baudouin and Clare, Mariana CA and Lessig, Christian and Maier-Gerber, Michael and Magnusson, Linus and others},
  journal={arXiv preprint arXiv:2406.01465},
  year={2024}
}
```

**APA:**

```apa
Lang, S., Alexe, M., Chantry, M., Dramsch, J., Pinault, F., Raoult, B., ... & Rabier, F. (2024). AIFS-ECMWF's data-driven forecasting system. arXiv preprint arXiv:2406.01465.
```

### Data Details

Describe the input and output data of your model here, use a table.

| Field                                                                                                                                                       | Level type                                                                   | Input/Output |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|--------------|
| Geopotential, horizontal and vertical wind components, specific humidity, temperature                                                                       | Pressure level: 50,100, 150, 200, 250,300, 400, 500, 600,700, 850, 925, 1000 | Both         |
| Surface pressure, mean sea-level pressure, skin temperature, 2 m temperature, 2 m dewpoint temperature, 10 m horizontal wind components, total column water | Surface                                                                      | Both         |

etc

### License

The model weights, and configuration files are published under a Creative Commons Attribution 4.0 International (CC BY 4.0).
To view a copy of this licence, visit https://creativecommons.org/licenses/by/4.0/

## Training

To train this model you can use the configuration files included in this repository and the following Anemoi packages:

```txt
anemoi-training==0.3.1
anemoi-models==0.4.0
anemoi-graphs>=0.4.4
```

### Datasets

TODO

### Training Step

Based on the different experiments we have made - the final training recipe for AIFS Single v1.0 has deviated slightly
from the one used for AIFS Single v0.2.1 since we found that we could get a well trained model by skipping the ERA5
rollout and directly doing the rollout on the operational-analysis (extended) dataset. When we say 'extended' we refer 
to the fact that for AIFS Single v0.2.1 we used just operational-analysis data from 2019 to 2021, while in this new 
release we have done the fine-tunning from 2016 to 2022. 

The other important change in the fine-tuning stage is that for AIFS Single v0.2.1 after the 6hr model training the
optimiser was not restarted (ie. rollout was done with the minimal lr of \\(3 × 10^{-7}\\)). For this release we have seen
that restarting the optimiser for the rollout improves the model's performance. For the operational-fine tuning rollout
stage, the learning rate cycle is restarted, gradually decreasing to the minimum value at the end of rollout.

- **Pre-training**: It was performed on ERA5 for the years 1979 to 2022 with a cosine learning rate (LR) schedule and a
total of 260,000 steps. The LR is increased from 0 to \\(10^{-4}\\) during the first 1000 steps, then it is annealed to a
minimum of \\(3 × 10^{-7}\\). The local learning rate used for this stage is \\(3.125 × 10^{-5}\\).
- **Fine-tuning**: The pre-training is then followed by rollout on operational real-time IFS NWP analyses for the years
2016 to 2022, this time with a local learning rate of \\(8 × 10^{−7}\\), which is decreased to \\(3 × 10^{−7}\\). Rollout steps
increase per epoch. In this second stage the warm up period of the optimiser is 100 steps to account for shorter length
of this stage. Optimizer step are equal to 7900 ( 12 epoch with ~630 steps per epoch).

As in the previous version of aifs-single for fine-tuning and initialisation of the model during inference, IFS fields
are interpolated from their native O1280 resolution (approximately \\(0.1°\\)) down to N320 (approximately \\(0.25°\\)).

#### Pretraining step

After creating the data, set the following environments variables and use the pretraining configuration file.

```bash
export DATASETS_PATH=???????
export OUTPUT_PATH=???????

cd training/pretraining
anemoi-training train --config-name=pretraining.yaml
```

#### Finetuning Step

Once pretraining is done, set the run id and use the finetuning configuration file.

```bash
export DATASETS_PATH=???????
export OUTPUT_PATH=???????
export PRETRAINING_RUN_ID=???????

cd training/finetuning
anemoi-training train --config-name=finetuning.yaml
```

This finetuning steps assumes an uninterrupted 13 epochs. If an error occurs, and training is restarted ensure you update
the `PRETRAINING_RUN_ID` to the new finetuning id. Additonally, the start of the `rollout` will need to be manually 
updated to ensure correct rollout finetuning.

#### Training Hyperparameters

- **Optimizer:** We use *AdamW* (Loshchilov and Hutter [2019]) with the \\(β\\)-coefficients set to 0.9 and 0.95.

- **Loss function:** The loss function is an area-weighted mean squared error (MSE) between the target atmospheric state
and prediction.

- **Loss scaling:** A loss scaling is applied for each output variable. The scaling was chosen empirically such that
all prognostic variables have roughly equal contributions to the loss, with the exception of the vertical velocities,
for which the weight was reduced. The loss weights also decrease linearly with height, which means that levels in 
the upper atmosphere (e.g., 50 hPa) contribute relatively little to the total loss value.

## Evaluation

AIFS is evaluated against ECMWF IFS (Integrated Forecast System) for 2022. The results of such evaluation are summarized in 
the scorecard below that compares different forecast skill measures across a range of
variables. For verification, each system is compared against the operational ECMWF analysis from which the forecasts
are initialised. In addition, the forecasts are compared against radiosonde observations of geopotential, temperature
and windspeed, and SYNOP observations of 2 m temperature, 10 m wind and 24 h total precipitation. The definition
of the metrics, such as ACC (ccaf), RMSE (rmsef) and forecast activity (standard deviation of forecast anomaly,
sdaf) can be found in e.g Ben Bouallegue et al. ` [2024].

### AIFS Single v1.0 vs AIFS Single v0.2.1 (2023)

<div style="display: flex; justify-content: center;">
  <img src="./assets/scorecard_single1.0_vs_single0.2.1_2023.png" alt="Scorecard comparing forecast scores of AIFS versus IFS (2022)" style="width: 80%;"/>
</div>

### AIFS Single v1.0 vs IFS (2024)

<div style="display: flex; justify-content: center;">
  <img src="./assets/scorecard_single1.0_vs_ifs_2024.png" alt="Scorecard comparing forecast scores of AIFS versus IFS (2022)" style="width: 80%;"/>
</div>

Forecasts are initialised on 00 and 12 UTC. The scorecard show relative score changes as function of lead time (day 1 to 10) for northern extra-tropics (n.hem),
southern extra-tropics (s.hem), tropics and Europe. Blue colours mark score improvements and red colours score
degradations. Purple colours indicate an increased in standard deviation of forecast anomaly, while green colours
indicate a reduction. Framed rectangles indicate 95% significance level. Variables are geopotential (z), temperature
(t), wind speed (ff), mean sea level pressure (msl), 2 m temperature (2t), 10 m wind speed (10ff) and 24 hr total
precipitation (tp). Numbers behind variable abbreviations indicate variables on pressure levels (e.g., 500 hPa), and
suffix indicates verification against IFS NWP analyses (an) or radiosonde and SYNOP observations (ob). Scores
shown are anomaly correlation (ccaf), SEEPS (seeps, for precipitation), RMSE (rmsef) and standard deviation of
forecast anomaly (sdaf, see text for more explanation).

Additional evaluation analysis including tropycal cyclone performance or comparison against other popular data-driven models can be found in AIFS preprint (https://arxiv.org/pdf/2406.01465v1) section 4.