import numpy as np
import os
import glob
import pandas as pd

CATCH_folders = os.listdir('RA_Jun2/')

for CATCH_folder in CATCH_folders:
	xlsx_file = glob.glob('*.xlsx')
	df = pd.read_excel(os.path.join('RA_Jun2/', CATCH_folder, xlsx_file[0]))

	unique_patient_id = np.unique(df['Patient ID'])
	for current_patient_id in unique_patient_id:
		patient_df = df.loc[df['Patient ID'] == current_patient_id]
		non_missing_df = patient_df[~patient_df['TOTAL SCORES E + J  '].str.contains('Missing', na=False)]

		image_names = os.listdir(os.path.join('RA_Jun2', CATCH_folder, current_patient_id))
