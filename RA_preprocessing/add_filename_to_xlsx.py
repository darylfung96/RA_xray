import pandas as pd
import os

all_tif_directory = 'all_RA_Jun2'
all_tif_files = os.listdir(all_tif_directory)

df = pd.read_excel('all_CATCH.xlsx')
for idx, row in df.iterrows():
	patient_id = row['Patient ID']
	timepoint = row['Timepoint']

	selected_tif_files = [tif_file for tif_file in all_tif_files if f'{patient_id}_{timepoint}' in tif_file]
	left_foot_filename = [selected_tif_file for selected_tif_file in selected_tif_files
	                      if 'feet' in selected_tif_file.lower() or ('foot' in selected_tif_file.lower() and 'left' in selected_tif_file.lower())]
	right_foot_filename = [selected_tif_file for selected_tif_file in selected_tif_files
	                      if 'feet' in selected_tif_file.lower() or (
				                      'foot' in selected_tif_file.lower() and 'right' in selected_tif_file.lower())]
	left_hand_filename = [selected_tif_file for selected_tif_file in selected_tif_files if
				                      'hand' in selected_tif_file.lower() and 'left' in selected_tif_file.lower()]
	right_hand_filename = [selected_tif_file for selected_tif_file in selected_tif_files if
				                      'hand' in selected_tif_file.lower() and 'right' in selected_tif_file.lower()]

	df.loc[idx, 'filename_LH'] = left_hand_filename[0] if len(left_hand_filename) else ''
	df.loc[idx, 'filename_RH'] = right_hand_filename[0] if len(right_hand_filename) else ''
	df.loc[idx, 'filename_LF'] = left_foot_filename[0] if len(left_foot_filename) else ''
	df.loc[idx, 'filename_RF'] = right_foot_filename[0] if len(right_foot_filename) else ''

df.to_excel('all_CATCH_with_filename.xlsx')

