import numpy as np
import pickle
import pandas as pd

'''
These functions are used in other programs.
'''

def fraction_lim(frame, frac_lim = 0):
	
	"""
	
	Input:
		frame = Pandas DataFrame at least containing column 'fraction'
		frac_lim = The lower limit of the fraction
	
	Output:
		Pandas DataFrame input frame containing only rows with fraction higher than frac_lim
	"""

	return frame.loc[frame.fraction > frac_lim].copy()
	
def percentile_cuts(frame, use_cols, low_cut = 5, high_cut = 95, verbose=True):

	"""
	Input:
		frame = Pandas DataFrame containing numeric columns
		use_cols = Numeric column names (list) that are included in the cut
		low_cut = Lower percentile cut limit
		high_cut = Upper percentile cut limit
	
	Output:
		Pandas DataFrame with only rows that fall outside percentile cuts for ALL columns
	"""

	percentiles_low = np.percentile(frame[use_cols], low_cut, axis=0) #5 percentile of every column
	percentiles_high = np.percentile(frame[use_cols], high_cut, axis=0) #95 percentile of every column
	
	perc_df = pd.DataFrame([percentiles_low, percentiles_high], columns = use_cols)
	
	#Remove all rows containing all nans in all use_cols
	temp = frame[use_cols].apply(lambda x: x[(x>perc_df.loc[0, x.name]) & (x < perc_df.loc[1, x.name])], axis=0).copy()
		
	frame_copy = frame.copy()
	result = frame_copy.loc[frame_copy.index.isin(temp.index)] #drop the dropped rows from original frame
	
	if verbose:
		print('Original DataFrame length: {}'.format(len(frame)))
		print('DataFrame length after percentile cuts: {}'.format(len(result)))
	
	return result
		

