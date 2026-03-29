AI Smart Battery Orchestration for Ghana SREP Mini-Grids

This project develops an AI system to manage battery storage in solar mini-grids
across Ghana's Scaling-Up Renewable Energy Programme (SREP). The system learns
when to charge and discharge a battery, hour by hour, to maximise electricity
supply to rural communities while protecting battery health.

It was trained on six years of real ERA5 climate data across six Ghana locations
and evaluated against a conventional rule-based controller.


SYSTEM SPECIFICATIONS

The trained model represents the average Ghana SREP mini-grid site.

Battery capacity      650 kWh LiFePO4
Solar PV              132.5 kWp (176.7 m2 panel area)
Community size        Approximately 1,318 people
Mean load             18.96 kW
Max charge rate       130 kW (0.2C)
SOC operating range   20% to 90%
Solar/Load ratio      1.51


TRAINING PIPELINE

The system is built in five sequential phases.

Phase 1   phase1_build_dataset.py
Loads ERA5 climate data for Tamale, Kumasi, and Axim. Merges with a Nigeria
national load profile scaled to the SREP community size. Outputs
master_dataset_scaled.csv.

Phase 2   phase2_train_lstm.py
Trains a two-layer LSTM with 128 hidden units to forecast the next 24 hours
of solar irradiance and load demand from an 8-feature, 24-hour lookback window.
Outputs best_lstm_scaled.pth.

Phase 3   phase3_environment.py
Defines a custom Gymnasium environment wrapping the LSTM forecaster. Models
battery physics with charge and discharge efficiency of 0.95, a solar
performance ratio of 0.75, and a three-mechanism LiFePO4 degradation model:
rainflow cycle aging (Xu et al. 2016), Arrhenius calendar aging with asymmetric
SOC stress (Wang et al. 2014), and a lithium plating penalty below 30% SOC.

Phase 4   phase4_ppo_training.py
Trains a PPO agent using Stable-Baselines3 over three curriculum phases
totalling 12 million steps. The observation space is 52 inputs covering SOC,
SOH, a 24-hour solar forecast, a 24-hour load forecast, hour, and month. The
action space is continuous from -1 (full discharge) to +1 (full charge).

Phase 5   phase5_evaluation.py
Evaluates the trained agent over a full six-year episode from January 2020 to
January 2026, totalling 52,608 hourly steps, across six Ghana locations
including three unseen during training: Accra, Bolgatanga, and Akosombo.
Results are compared against a rule-based controller.


RESULTS (TAMALE, 6 YEARS)

                        Rule-Based      AI (PPO)      Change
Load Served               57.5%          96.6%        +39.1 pp
Loss of Load Prob         43.5%           7.9%        -35.6 pp
Energy Not Served     422,339 kWh      33,825 kWh       -92%
Battery Lifespan        6.9 years       9.7 years      +2.8 years
Final SOH (6yr)           82.6%          87.6%         +5.0 pp


APPLICATIONS

minigrid_app_v3.py   Research Dashboard

Designed for researchers and engineers. Offers two modes. Live Weather fetches
real hourly solar irradiance from Open-Meteo, runs a 24-hour simulation, and
shows an AI decision explainer. Historical Results displays pre-computed
six-year evaluation results for all six locations with SOH trajectory charts
and KPI comparisons.

Run with:   streamlit run notebooks/minigrid_app_v3.py


minigrid_operator_app.py   Operator Tool

Designed for Ghana mini-grid field technicians. The operator enters their
location, battery size, solar panel size, daily community load, and current
battery level. The app fetches live weather and produces a 24-hour
hour-by-hour plan in plain English with blackout risk alerts.

Run with:   streamlit run notebooks/minigrid_operator_app.py


DATA SOURCES

ERA5 CDS (ECMWF)    Hourly solar irradiance, temperature, and precipitation
                    for six Ghana locations from 2020 to 2026

Nigeria Load Data   Hourly national demand profile scaled to SREP community
                    size of approximately 1,318 people and 18.96 kW mean load

Open-Meteo          Live weather for the applications, free with no API key
                    required

Ghana SREP          System specifications from AfDB and World Bank programme
                    documents covering 35 mini-grids and 4.525 MWp total


INSTALLATION

pip install torch stable-baselines3 gymnasium streamlit plotly pandas numpy scikit-learn requests


FOLDER STRUCTURE

data         ERA5 climate CSVs, master dataset, evaluation results
models       Trained LSTM weights, PPO agent, VecNormalize statistics
notebooks    All pipeline scripts and application files


LIMITATIONS

The trained agent is specific to LiFePO4 battery chemistry. Lead-acid systems
are not supported. The operator app uses proportional scaling to handle
different system sizes, which is a heuristic approach since the agent was
trained on a fixed 650 kWh system and does not genuinely generalise to
arbitrary configurations. No diesel generator backup is modelled. The agent
generalises across six Ghana locations but is not a universal battery
management system.


NEXT STEPS

The next phase will retrain the agent with system parameters, specifically
battery capacity in kWh, solar size in kWp, and mean load in kW, included in
the observation space. This will allow the agent to genuinely learn to manage
any Ghana SREP mini-grid size rather than relying on heuristic scaling.


REFERENCES

Xu, B. et al. (2016). Modeling of lithium-ion battery degradation for cell
life assessment. IEEE Transactions on Smart Grid.

Wang, J. et al. (2014). Cycle-life model for graphite-LiFePO4 cells.
Journal of Power Sources.

Hersbach, H. et al. (2020). The ERA5 global reanalysis. Quarterly Journal
of the Royal Meteorological Society.

African Development Bank. Ghana SREP Mini-grid and Solar PV Net Metering
Project Appraisal Report (2022).
