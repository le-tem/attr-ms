# Anthropogenic Climate Change Amplifies MS Symptom Risk in Austria

Quantify the impact of climate change on core temperature increases that elicit Uhthoff’s phenomenon *(Temple et al., submitted)*

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Project Structure](#project-structure)  
3. [Prerequisites](#prerequisites)  
4. [Usage](#usage)  
5. [Data Sources](#data-sources)  
6. [Dependencies](#dependencies)  
7. [Contact](#contact)

## Project Overview
This repository contains analyses and visualizations quantifying how anthropogenic climate change may amplify the risk of temperature-induced multiple sclerosis symptoms in Austria.

## Project Structure
`/main`  
  Contains all analysis scripts for:
  1. Climate data analysis and attribution  
  2. [physiological_modeling](main/physiological_modeling): estimates the core temperature rise 
  3. Visualizations, statistical analyses, and result tables  

## Prerequisites
- Python 3.8+  
- Required Python packages listed in `Pipfile` or `requirements.txt` in each subdirectory

## Usage
1. Clone the repository  
   ```bash
   git clone https://github.com/le-tem/attr-ms.git
   cd attr-ms
   ```  
2. Install dependencies, for example using pipenv:
   ```bash
   cd main/physiological_modeling
   pipenv install --dev
   ```  
3. Run the main analysis  
   ```bash
   python main/physiological_modeling/analysis.py
   ```  

## Data Sources
All climate data are publicly accessible:
- **ZAMG Observations**: Temperature & relative humidity  
  https://data.hub.zamg.ac.at/dataset/klima-v1-1h (last accessed 20.06.2025)  
- **MERRA-2 Reanalysis**: NASA GES DISC  
  https://disc.gsfc.nasa.gov/datasets/M2TMNXSLV_5.12.4/summary (last accessed 23.07.2025)  
- **CMIP6 Simulations**: CEDA Archive  
  https://catalogue.ceda.ac.uk/?q=CMIP6&sort_by=relevance&results_per_page=20 (last accessed 23.07.2025)  

## Dependencies
Physiological modeling relies on **pythermalcomfort**:  
https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort

## Contact
Lucy Temple: lucy.temple@unibe.ch
```