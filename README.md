# Corrective Wind Smoke
This codes consists of two parts for analyzing and investigating of corrective approaches for uncertain wind impacts on smoke model performance. 

## Wind bias reduction method
Aimed at minimizing wind bias in meteorological simulations
* Nudging Benchmark (NB): Nudging using collected data
* Augmentation Nudging (AN): Nudging using interpolated data
* Initial Condition and Boundary Condition (ICBC): Adjustments to initial and boundary conditions for wind bias reduction

## Smoke Model Evaluation Methods
Quantitatively assess wind impacts on smoke simulations
* Rotation and Translation (RT): Rotate the monitor location with burn units as center or translate the distance between monitor location and burn units.
* Equal Time Back/forward Trajectory (ETBFT): Combining backward trajectory and forward trajectory under same time to find pseudo-monitor locations.
* Equal Distance Back/forward Trajectory (EDBFT): Combining backward and forward trajectories with the same transport distance but at different times to determine pseudo-monitor locations.

## Environment Set Up
```conda env create -f environment.yml```

## Code
### LittleRGenerator:
The provided codes can create Little R format data based on provided CSV format wind measurement data. The provided CSV format file should include the following columns:
| UTC_time            | monitor       | lon     | lat     | elevation | wdspd  | wddir  |
|---------------------|--------------|---------|---------|-----------|--------|--------|
| YYYY-MM-DD HH:MM:SS | Monitor Name | Longitude | Latitude | Elevation | Wind speed  | Wind direction  |

Units: 
* Longitude: float in degrees
* Latitude: float in degrees
* Elevation: above ground (m)
* Wind speed: m/s
* Wind direction: degrees

Process the wind measurement data and run `StandardCSVLittleR.py` to generate the Little R format data for FDDA.

## AN:
1. `AugNudgeProcess.py`: The code uses wind data in CSV format to generate an interpolated wind field on a WRF-defined grid. The elevation is provided by METCRO2D but can also be obtained from datasets like NLCD for future implementations. The output file is in a standard wind CSV format, which is later used to create Little R format data by running `StandardCSVLittleR.py`.

## ICBC:
1. `ReviseMetGrid.py`: the code is used to rotate or scale the wind field in metgrid file to reduce the bias between ICBC and measurements.
2. `ReviseWRF.py`: the code is used to rotate or scale the wind field in WRF file to reduce the bias between ICBC and measurements. The WRF file will be used in ndown.exe program to provide a adjusted ICBC for fire model simulations.

## RotationTraslation:
1. `ShadedConcentration.py`: the code uses the RT method to calculate the corrected concentration and its uncertainty for each monitor and each dates. It will create a pickle format file for visulizations (`Visualization/ConcUncertainty.py`).
2. `SensitivityAnalysis.py`: the code provides sensitivity analysis on impacts from biased wind speed and wind direction on concentration simulations based on RT method.

## Trajectory
1. `EqualTimeTrajSFIRE.py`: the code uses the ETBFT method to calculate the corrected concentration and its uncertainty for each monitor and each dates. It will create a pickle format file for visulizations (`Visualization/ConcUncertainty.py`).
2. `EqualDistTrajSFIRE.py`: the code uses the EDBFT method to calculate the corrected concentration and its uncertainty for each monitor and each dates. It will create a pickle format file for visulizations (`Visualization/ConcUncertainty.py`).

## Utilities
1. `metrics.py`: Statistical metrics for model performance evaluation.
2. `uncertainty_util.py`: implementations for RT, ETBFT, and EDBFT methods which are wrapped in functions.
3. `util.py`: utilities for supporting analysis or get WRF grid informations. For other CTMs, consider to revise the functions here such as `MetGridInfo` or `WRFGridInfo`.
4. `visualization_util.py`: utilities for visualizations.