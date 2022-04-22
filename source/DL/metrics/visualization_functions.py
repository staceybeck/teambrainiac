


def find_best_worst_index(test):
  '''
  Finds the best and worst indicies in a test set based on accuracies 
  '''
  best_acc = 0
  worst_acc = 1
  best_index = None
  worst_index = None

  for i,acc in enumerate(test['epoch_1']['accuracy']):
    acc = acc.item()
    if acc <= worst_acc:
      worst_acc = acc
      worst_index = i
    if acc >= best_acc:
      best_acc = acc
      best_index = i
  return best_index, worst_index





def process_predictions(test, prediction_index):
  '''
  Creates predictions dataframe from test dictionary
  '''
  preds_df = pd.DataFrame(test['preds_'+str(prediction_index)], columns=['Down Regulation', 'Up Regulation'])

  preds_df['Time'] = preds_df.index
  preds_df['Prediction'] = [x.item() for x in preds_df['Up Regulation']]
  scaler = MinMaxScaler(feature_range=(-1,1))
  preds_df['Prediction'] = scaler.fit_transform(preds_df[['Prediction']])

  preds_df['True Label'] = [x.item() if x == 1 else -1 for x in test['labels']]

  pred_label = []
  for down_pred, up_pred in zip(preds_df['Down Regulation'], preds_df['Up Regulation']):
    if down_pred > up_pred:
      pred_label.append(-1)
    elif down_pred < up_pred:
      pred_label.append(1)
  preds_df['Predicted Label'] = pred_label

  preds_df = preds_df.drop(['Up Regulation', 'Down Regulation'], axis=1)

  return preds_df







def plot_decisions(ds, labels, time, title, run):
    """
    :param ds:
    :param labels:
    :param time:
    :param title:
    :param run:
    :return:
    """
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(ds,
            label='scaled CNN score\n',
            color="black"
            )
    ax.plot(labels,
            label='scaled true label values\n',
            color="#446ccf"
            )
    ax.set_xlabel('\nTime',
                  fontsize=14
                  )
    ax.set_ylabel('Prediction', fontsize=14)
    ax.axhline(0.0,
               ls="dotted",
               color='#b62020',
               label="> 0.0 Up-Regulating \n\n< 0.0 Down-Regulating"
               )
    ax.axvline(0,
               ls="dotted",
               color='gray'
               )

    
    ax.set_xticklabels([""])

    ax.set_title(title, fontsize=20)
    lgd = ax.legend(loc=(1.01, 0.5), fontsize=14)
    return plt.tight_layout()
