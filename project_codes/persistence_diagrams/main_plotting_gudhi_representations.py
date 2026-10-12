import os
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.rcParams['text.usetex'] = False 

from gudhi_python_codes.gudhi_plotting_class import GudhiPlottingClass
from gudhi_python_codes.composite_builder import CompositePipeline 
from gudhi_python_codes.betti_numbers_manager import BettiNumbersManager
from gudhi_python_codes.covariates_vs_persistence import CovariatePersistencePipeline



if __name__ == "__main__":

    beginYear, endYear = 1945, 2010
    years = [1955,1975,1995,2010]
    n_years = list(range(beginYear,endYear+1))
    cases = [n_years.index(year) for year in years]

    all_covariates = ['logoil', 'ln_gdppc', 'lnpop', 'muslim', 
                      'borders', 'polity','revleader', 'new_oil_state', 
                      'coldwar', 'majorpower']
    tda_features = ["Amplitude","Entropy","L1","C1","L2","NumberOfPoints"]
    homology_dimensions = [0,1,2,3,4]
    hmax = len(homology_dimensions) - 1
    
    folder = f'../../project_results/gudhi_multiple_tda_results_H_{hmax}/'
    images = os.path.join(folder, 'images')

    data_folder='../../project_data/'

    images = os.path.join(folder, 'images/')
    PersistenceDiagrams = os.path.join(folder, 'PersistenceDiagrams/')
    PersistenceLandscapes = os.path.join(folder, 'PersistenceLandscapes/')
    for sub in [folder, images, PersistenceDiagrams, PersistenceLandscapes]: 
        os.makedirs(sub, exist_ok=True)

    args = {
        'beginYear': beginYear,
        'endYear': endYear,
        'n_comp': 3,
        'years': years,
        'n_years': n_years,
        'cases': cases,
        'homology_dimensions': homology_dimensions,
        'max_dimension':  hmax,
        'folder': folder, 
        'images': images,
        "PersistenceLandscapes": PersistenceLandscapes,
        "PersistenceDiagrams":PersistenceDiagrams,
        'data_folder':data_folder,
        'tda_features': tda_features,
        'all_covariates': all_covariates,
        'C': 1.0,
        'p': 1.0,
        'bins': 50,
        'n_components': 3,
        "max_edge_length": 6.,
        'homology_coeff_field':11,
        'min_persistence': 0.0,    
        'projection_dim':2,
        "min_persistence": 0.,     
        "top_n_per_dim": 10,
        "projection_dim": 2,
        "homology_coeff_field": 11,
        "max_points": 2000,
        "neighbor_k": 32,
        "radius_quantile": 20.0,    
        }
    
  
    plot_class = GudhiPlottingClass(args)
    plot_class.Plotting_Persistence_Diagrams()
    plot_class.Plotting_Persistence_Barcodes()
    plot_class.Plotting_Entropy()
    plot_class.Plotting_Amplitude()
    plot_class.Plotting_NumberOfPoints()
    plot_class.Plotting_landscapes()
    plot_class.plot_l2_series()
 
    plot_class.plot_c1_series()
    plot_class.plot_c1_l1_series()
    plot_class.plot_c1_l1_l2()
    plot_class.plot_covariates_overlaid_with_composite_pca1()
    plot_class.plot_midpoint_hist()

    pipeline = CompositePipeline(args)
    pipeline.build_and_save_composites(mode="separate")

    bn = BettiNumbersManager(args)
    bn.global_scale_pds() 
    bn.betti_numbers() 
    bn.betti_summary() 
    bn.betti_timeseries() 
    bn.betti_heatmap() 
    bn.betti_peaks_and_auc()
    

    wc = CovariatePersistencePipeline(args)
    wc.run_pipeline()


    
