from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob
import os

import pandas as pd

patient_part_to_correction = {
    'fooleft': 'footleft',
    'fooright': 'footright',
    'footlefrt': 'footleft',
    'footrightf': 'footright',
    'left foot': 'footleft',
    'left hand': 'handleft',
    'leftfood': 'footleft',
    'leftfoot': 'footleft',
    'lefthand': 'handleft',
    'leftthand': 'handleft',
    'right foot': 'footright',
    'right hand': 'handright',
    'rightfoot': 'footright',
    'righthand': 'handright',

    'footleft': 'footleft',
    'footright': 'footright',
    'feet': 'feet',
    'hands': 'hands',
    'handright': 'handright',
    'handleft': 'handleft'
}

for folder in os.listdir('./RA_Jun2/'):
    patient_parts = []
    dates = []

    if not os.path.isdir(os.path.join('RA_Jun2', folder)):
        continue

    for file in glob.glob(f"./RA_Jun2/{folder}/**/*.tif", recursive=True):
        patient_part = patient_part_to_correction[file.split("_")[-1].replace(".tif", "").lower()]
        date = file.split(os.path.sep)[-1].split("_")[1]

        patient_parts.append(patient_part)
        dates.append(date)

    unique_parts = np.unique(patient_parts)
    unique_dates = np.unique(dates)

    # summmary information
    total_num_images = 0
    total_num_timepoints = len(unique_dates)
    total_num_patients = 0
    total_num_leftfoot = 0
    total_num_rightfoot = 0
    total_num_lefthand = 0
    total_num_righthand = 0

    patient_list = []
    all_patients = {}
    parts_dict = OrderedDict()
    parts_dict['patient_id'] = ''
    for date in unique_dates:
        for part in unique_parts:
            parts_dict[f'{date}__{part}'] = 0

    for patient_file in glob.glob(f"./RA_Jun2/{folder}/**/*.tif", recursive=True):
        total_num_images += 1
        patient_id = patient_file.split(os.path.sep)[-2]
        patient_part =patient_part_to_correction[patient_file.split('_')[-1].replace(".tif", "").lower()]
        patient_date = patient_file.split(os.path.sep)[-1].split("_")[1]

        # calculate the number of count for each joint
        if all_patients.get(patient_id, None) is None:
            current_patient = parts_dict.copy()
            if current_patient['patient_id'] == '':
                current_patient['patient_id'] = patient_id

            all_patients[patient_id] = current_patient
            all_patients[patient_id][f'{patient_date}__{patient_part}'] += 1
        else:
            all_patients[patient_id][f'{patient_date}__{patient_part}'] += 1

    for patient_id, patient_info in all_patients.items():
        total_num_patients += 1
        patient_info['patient_id'] = patient_id
        patient_list.append(patient_info)

    catch_dataframe = pd.DataFrame(patient_list)

    # calculate how many lefts/rights
    catch_dataframe['handleft'] = ''
    catch_dataframe['handright'] = ''
    catch_dataframe['footleft'] = ''
    catch_dataframe['footright'] = ''
    catch_dataframe['missing_handleft'] = ''
    catch_dataframe['missing_handright'] = ''
    catch_dataframe['missing_footleft'] = ''
    catch_dataframe['missing_footright'] = ''    # check parts from patients how many => (only left, only right, left and right)
    for patient_id in catch_dataframe['patient_id']:
        current_patient = catch_dataframe.loc[catch_dataframe['patient_id'] == patient_id]

        # how many left foot
        patient_cols = [col for col in current_patient.columns if 'footleft' in col or 'feet' in col]
        patient_leftfoot = catch_dataframe.loc[catch_dataframe['patient_id'] == patient_id, patient_cols]
        num_left_foot = sum(filter(None, patient_leftfoot.values.ravel().tolist()))
        missing_num_left_foot = len([value for value in patient_leftfoot.values.ravel() if value == 0])
        catch_dataframe.loc[catch_dataframe['patient_id'] == patient_id, 'footleft'] = num_left_foot
        catch_dataframe.loc[catch_dataframe['patient_id'] == patient_id, 'missing_footleft'] = missing_num_left_foot
        total_num_leftfoot += num_left_foot

        # how many right foot
        patient_cols = [col for col in current_patient.columns if 'footright' in col or 'feet' in col]
        patient_rightfoot = catch_dataframe.loc[catch_dataframe['patient_id'] == patient_id, patient_cols]
        num_right_foot = sum(filter(None, patient_rightfoot.values.ravel().tolist()))
        missing_num_right_foot = len([value for value in patient_rightfoot.values.ravel() if value == 0])
        catch_dataframe.loc[catch_dataframe['patient_id'] == patient_id, 'footright'] = num_right_foot
        catch_dataframe.loc[catch_dataframe['patient_id'] == patient_id, 'missing_footright'] = missing_num_right_foot
        total_num_rightfoot += num_right_foot

        # how many left hand
        patient_cols = [col for col in current_patient.columns if 'handleft' in col or 'hands' in col]
        patient_lefthand = catch_dataframe.loc[catch_dataframe['patient_id'] == patient_id, patient_cols]
        num_left_hand = sum(filter(None, patient_lefthand.values.ravel().tolist()))
        missing_num_left_hand = len([value for value in patient_lefthand.values.ravel() if value == 0])
        catch_dataframe.loc[catch_dataframe['patient_id'] == patient_id, 'handleft'] = num_left_hand
        catch_dataframe.loc[catch_dataframe['patient_id'] == patient_id, 'missing_handleft'] = missing_num_left_hand
        total_num_lefthand += num_left_hand

        # how many right hand
        patient_cols = [col for col in current_patient.columns if 'handright' in col or 'hands' in col]
        patient_righthand = catch_dataframe.loc[catch_dataframe['patient_id'] == patient_id, patient_cols]
        num_right_hand = sum(filter(None, patient_righthand.values.ravel().tolist()))
        missing_num_right_hand = len([value for value in patient_righthand.values.ravel() if value == 0])
        catch_dataframe.loc[catch_dataframe['patient_id'] == patient_id, 'missing_handright'] = missing_num_right_hand
        total_num_righthand += num_right_hand


    # check dates  how many => (only left, only right, left and right)
    dates_info = []
    for date in unique_dates:
        footleft = catch_dataframe[f'{date}__footleft'].values.sum()
        footright = catch_dataframe[f'{date}__footright'].values.sum()
        handleft = catch_dataframe[f'{date}__handleft'].values.sum()
        handright = catch_dataframe[f'{date}__handright'].values.sum()

        date_missing_footleft = len([value for value in catch_dataframe[f'{date}__footleft'].values.ravel() if value == 0])
        date_missing_footright = len([value for value in catch_dataframe[f'{date}__footright'].values.ravel() if value == 0])
        date_missing_handleft = len([value for value in catch_dataframe[f'{date}__handleft'].values.ravel() if value == 0])
        date_missing_handright = len([value for value in catch_dataframe[f'{date}__handright'].values.ravel() if value == 0])

        date_info = {
            'date': date,
            'footleft': footleft,
            'footright': footright,
            'handleft': handleft,
            'handright': handright,
            'missing_footleft': date_missing_footleft,
            'missing_footright': date_missing_footright,
            'missing_handleft': date_missing_handleft,
            'missing_handright': date_missing_handright
        }
        dates_info.append(date_info)

    catch_dataframe.to_excel(f'{folder}_RA_files_info.xlsx')

    # plot patients
    all_cols = ['handleft', 'handright', 'footleft', 'footright', 'missing_handleft',
                'missing_handright', 'missing_footleft', 'missing_footright']
    plt.clf()
    for col in all_cols:
        plt.plot(catch_dataframe['patient_id'], catch_dataframe[col], label=col)
    plt.legend()
    plt.rcParams.update({'font.size': 8})
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.savefig(f'{folder} patient.png')


    # plot dates
    dates_df = pd.DataFrame(dates_info)
    plt.clf()
    for col in dates_df.columns[1:]:
        plt.plot(dates_df['date'], dates_df[col], label=col)
    plt.legend()
    plt.rcParams.update({'font.size': 6})
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.savefig(f'{folder} date.png')

    dates_df.to_excel(f'{folder}_dates_RA_files_info.xlsx')


    # plot histogram
    total_data = pd.DataFrame({'total': ['total_num_images', 'total_num_timepoints', 'total_num_patients',
                    'total_num_leftfoot', 'total_num_rightfoot', 'total_num_lefthand', 'total_num_righthand'],
                  'values': [total_num_images, total_num_timepoints, total_num_patients, total_num_leftfoot,
                             total_num_rightfoot, total_num_lefthand, total_num_righthand]})
    plt.clf()
    ax = sns.barplot(x='values', y='total', data=total_data, color='blue')
    ax.set_xlabel('total')
    ax.set_ylabel('numbers')
    ax.set_yticklabels(['total_num_images', 'total_num_timepoints', 'total_num_patients',
                    'total_num_leftfoot', 'total_num_rightfoot', 'total_num_lefthand', 'total_num_righthand'])
    plt.autoscale()
    ax.figure.savefig(f'{folder} histgram.png', bbox_inches='tight')

    # get the total sum
    catch_dataframe.loc[:, 'patient_total'] = catch_dataframe.sum(numeric_only=True, axis=1)
    catch_dataframe.loc['total'] = catch_dataframe.sum(numeric_only=True, axis=0)
    catch_dataframe.to_excel(f'{folder}_RA_files_info.xlsx')


# num patients
# total images
# num left hand
# num right hand
# num left foot
# num right foot
# num timepoints
