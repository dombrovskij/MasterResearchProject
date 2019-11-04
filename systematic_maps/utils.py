import os

def fig_dir():
    '''
    figure directory
    '''
    fig_dir = os.path.dirname(os.path.realpath(__file__))+'/graphs/'
    
    return fig_dir

def dat_dir():
    '''
    Dat directory
    '''
    dat_dir = '/disks/shear12/dombrovskij/systematic_maps/data'

    return dat_dir
