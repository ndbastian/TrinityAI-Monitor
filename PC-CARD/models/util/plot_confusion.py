import matplotlib.pyplot as plt
import numpy  as np
from sklearn.metrics import confusion_matrix
import itertools
#%%

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          savefilename=""):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      cm = cm * 100
      print("\nNormalized confusion matrix")
  else:
      print('\nConfusion matrix, without normalization')

  if normalize:
      np.set_printoptions(precision=2)  # set NumPy to 2 decimal places
  print(cm)
  print ()
  fig, ax = plt.subplots(figsize=(20, 20), dpi=250)
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  tick_marks = np.arange(len(classes))

  plt.xticks(tick_marks, classes, rotation=90)
  plt.yticks(tick_marks, classes)

  fmt = '.0f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.title(title)
  plt.colorbar(ax=ax)
  plt.tight_layout()
  if savefilename:
    plt.savefig(savefilename)



