
import pandas as pd
import os

from gudhi_python_codes.gudhi_persistence_representations_class import GudhiPersistenceRepresentationsClass
from gudhi_python_codes.gudhi_persistence_diagrams_class import GudhiPersistenceDiagramsClass


if __name__ == "__main__":

    homology_dimensions = [0,1,2,3,4]
    beginYear, endYear = 1945, 2010
    hmax = len(homology_dimensions) - 1
    
    folder = f'../../project_results/gudhi_multiple_tda_results_H_{hmax}/'
    images = os.path.join(folder, 'images')
    PersistenceDiagrams = os.path.join(folder, 'PersistenceDiagrams')
    PersistenceLandscapes = os.path.join(folder, 'PersistenceLandscapes')
  
    for sub in [folder, images, PersistenceDiagrams, PersistenceLandscapes]: 
        os.makedirs(sub, exist_ok=True)

    args = {
        'beginYear': beginYear,
        'endYear':endYear,
        'C': 5. ,
        'p': 3,
        'bins': 100, 
        'max_edge_length': 12.0,
        'max_dimension':  hmax,             
        'n_components':  3,
        'beginYear':  beginYear,
        'endYear':  endYear,
        'n_years': range(beginYear, endYear+1),
        'data_folder': '../../project_data/',
        'homology_dimensions':  homology_dimensions,
        'folder': folder,
        'images': images,
        'tda_features': ["Amplitude","Entropy","L1","C1","L2","NumberOfPoints"],
        "PersistenceLandscapes": PersistenceLandscapes,
        "PersistenceDiagrams":PersistenceDiagrams,
    }


    """
    Comoputing and Saving PDs
    """
    gudhi_PD_class = GudhiPersistenceDiagramsClass(args)
    gudhi_PD_class.grouping_PDs()
    print('Done computing persistence diagrams for all years!')

    

    """
    Computing representations and saving
    """
    gudhi_rep_class = GudhiPersistenceRepresentationsClass(args)
    gudhi_rep_class.persistence_entropy()
    gudhi_rep_class.amplitude()
    gudhi_rep_class.number_of_points()
    gudhi_rep_class.landscape_l2()
    gudhi_rep_class.compute_c1()
    gudhi_rep_class.betti_numbers()
    gudhi_rep_class.betti_summary()
    print('Done computing persistence  entropy , amplitude, number of points , landscapes for all years!')

  
