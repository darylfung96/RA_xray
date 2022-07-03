import pandas as pd
import numpy as np
import os
import glob

processed_xlsx_folder = 'processed_xlsx_RA_files'
xlsx_folder = 'xlsx_RA_files/RA_Jun2'
os.makedirs(processed_xlsx_folder, exist_ok=True)

for xlsx_file in glob.glob(f'{xlsx_folder}/**/*.xlsx'):
	df = pd.read_excel(xlsx_file)

	patient_ids = np.unique(np.array(df['Patient ID']))
	timepoints = np.unique(np.array(df['Timepoint']))

	# intiialize columns
	column_dict = {'Patient ID': None}
	for col in df.columns[2:]:
		for timepoint in timepoints:
			column_dict[f'{col}__{timepoint}'] = 'Missing'

	all_patients = []
	for patient_id in patient_ids:
		patient_df = df.loc[df['Patient ID'] == patient_id]
		patient_dict = column_dict.copy()

		patient_dict['Patient ID'] = patient_id
		for timepoint in np.unique(patient_df['Timepoint']):
			current_patient_timepoint_df = patient_df.loc[patient_df['Timepoint'] == timepoint]
			for col in df.columns[2:]:
				patient_dict[f'{col}__{timepoint}'] = current_patient_timepoint_df[col].item()

		all_patients.append(patient_dict)

	pd.DataFrame(all_patients, columns=column_dict.keys()).to_excel(f'{processed_xlsx_folder}/{os.path.split(xlsx_file)[-1]}')