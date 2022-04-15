#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    dataframes.py
# @Author:      staceyrivet
# @Time:        4/14/22 12:10 PM
# @IDE:         PyCharm


def model_metric_data(mask_type):
    """

    :param mask_type:
    :return:
    """


  # Define directories where metrics and models are stored on AWS
  metric_directory = f'metrics/group_svm/{mask_type}/'
  model_directory = 'models/group/'

  # Assign path names to variables for Adolescent and Young Adult
  ad_metric_file = f'AD_AD_[2, 3]_{mask_type}_metrics.pkl'
  ad_model_file = f'AD_AD_[2, 3]_{mask_type}_X_y_model.pkl'
  ya_metric_file = f'YA_YA_[2, 3]_{mask_type}_metrics.pkl'
  ya_model_file = f'YA_YA_[2, 3]_{mask_type}_X_y_model.pkl'

  # Grab Adolescent and Young Adult metric and model dictionaries
  adad_metric = access_load_data(f"{metric_directory}{ad_metric_file}", False)
  adad_model = access_load_data(f"{model_directory}{ad_model_file}", False)
  yaya_metric = access_load_data(f"{metric_directory}{ya_metric_file}", False)
  yaya_model = access_load_data(f"{model_directory}{ya_model_file}", False)

  # Define empty dictionary
  df_dict = defaultdict(list)

  # Load the data and save as a DF
  mask_data_list = ((mask_type, adad_model, adad_metric, "Adolescent"), (mask_type, yaya_model, yaya_metric,"Young Adult"))

  for msk, mdl, mtric, grp in mask_data_list:
    df_dict['mask_type'].append(msk)
    df_dict['group'].append(grp)
    df_dict['model'].append(mdl['model'][0])
    df_dict['X'].append(mdl['X_train'][0])
    df_dict['y'].append(mdl['y_train'][0])
    df_dict['y_v'].append(mtric['y_v'][0])
    df_dict['y_t'].append(mtric['y_t'][0])
    df_dict['val_preds'].append(mtric['val_preds'][0])
    df_dict['test_preds'].append(mtric['test_preds'][0])
    df_dict['val_probs'].append(mtric['val_probs'][0])
    df_dict['test_probs'].append(mtric['test_probs'][0])


  df = pd.DataFrame(df_dict)
  df.to_csv(f'/content/gdrive/MyDrive/YA_AD_{msk}_df.csv', index = False, header = True)

  return True



def combine_all_group_df(mask_var):
    """
    :param mask_var:
    :return:
    """

  for data in mask_var:
    print(f"Loading {data} data...")

    path = f"/content/gdrive/MyDrive/YA_AD_{data}_df.csv"

    if data == 'mask':
      df1 = pd.read_csv(path)
    else:
      df2 = pd.read_csv(path)
      df1 = pd.concat([df1, df2])

  df1.to_csv(f'/content/gdrive/MyDrive/YA_AD_eachmask_df.csv', index = False, header=True)

  return df1




if __name__ == "__main__":
    # Create the dataframes per mask type and store locally
    mask_var = ['mask' ,'masksubACC', 'masksubAI', 'masksubNAcc', 'masksubmPFC']
    for data in mask_var:
      print(f"Creating dataframe for mask type: {data}")
      model_metric_data(data)

    # Combine all dataframes
    combine_all_group_df(mask_var)