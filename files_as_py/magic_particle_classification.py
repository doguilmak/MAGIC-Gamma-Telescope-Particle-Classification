# -*- coding: utf-8 -*-
"""MAGIC_particle_classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15lw98GGAHK7CAdoSMeaJdaXsbXfUnRzN

<h1  align=center><font  size = 6>Classify High Energy Gamma Particles Observed From MAGIC Telescope</font></h1>

<p align="center">
    <img src="https://www.iac.es/sites/default/files/styles/crop_cinemascope_48_17_to_1920px/public/images/installation/Perfil%20ORM%20V%C3%ADa%20Lactea.jpg?h=ef71438a&itok=VvicsMzE" height=370  width=1000> 
</p>

<small>Picture Source: <a href="https://www.iac.es/en/observatorios-de-canarias/telescopes-and-experiments/magic-telescopes">Instituto de Astrofísica de Canarias</a></small>

<br>

<h2>Data Set Information</h2>

<p>The data are <i>MC generated</i> (see below) to simulate registration of high energy gamma particles in a ground-based atmospheric Cherenkov gamma telescope using the imaging technique. Cherenkov gamma telescope observes high energy gamma rays, taking advantage of the radiation emitted by charged particles produced inside the <i>electromagnetic showers</i> initiated by the <i>gammas</i>, and <i>developing</i> in the <i>atmosphere</i>. This <i>Cherenkov radiation (of visible to UV wavelengths)</i> leaks through the <i>atmosphere</i> and gets recorded in the detector, allowing reconstruction of the <i>shower parameters</i>. The available information consists of pulses left by the incoming <i>Cherenkov photons</i> on the <i>photomultiplier tubes</i>, arranged in a plane, the camera. Depending on the energy of the primary <i>gamma</i>, a total of few hundreds to some <i>10000 Cherenkov photons</i> get collected, in <i>patterns (called the shower image)</i>, allowing to discriminate statistically those caused by primary <i>gammas (signal)</i> from the images of hadronic showers initiated by cosmic rays in the <i>upper atmosphere (background)</i>.</p>

<p>Typically, the image of a shower after some <i>pre-processing</i> is an <i>elongated cluster</i>. Its long axis is oriented towards the camera center if the <i>shower axis</i> is parallel to the telescope's <i>optical axis</i>, i.e. if the telescope axis is directed towards a point source. A <i>principal component analysis</i> is performed in the camera plane, which results in a correlation <i>axis</i> and defines an <i>ellipse</i>. If the <i>depositions</i> were distributed as a <i>bivariate Gaussian</i>, this would be an <i>equidensity ellipse</i>. The characteristic parameters of this <i>ellipse (often called Hillas parameters)</i> are among the image parameters that can be used for <i>discrimination</i>. <i>The energy depositions</i> are typically <i>asymmetric</i> along the <i>major axis</i>, and this <i>asymmetry</i> can also be used in <i>discrimination</i>. There are, in addition, further discriminating characteristics, like the extent of the cluster in the image plane, or the total sum of depositions. This information has taken from <a href='https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope'>archive.ics.uci.edu</a> site.</p>

<br>

<h2>Keywords</h2>
<ul>
  <li>Neural Networks</li>
  <li>Space</li>
  <li>Telescope</li>
  <li>Classification</li>
  <li>High Energy Gamma Particles</li>
	<li>Deep Learning</li>
</ul>

<br>

<p>The data set was generated by a <i>Monte Carlo program, Corsika</i>, described in:
D. Heck et al., <i>CORSIKA</i>, A Monte Carlo code to <i>simulate extensive air showers</i>, <i><a href='http://rexa.info/paper?id=ac6e674e9af20979b23d3ed4521f1570765e8d68'>Forschungszentrum Karlsruhe FZKA 6019 (1998)</a></i>.

The program was run with parameters allowing to observe events with energies down to below <i>50 GeV</i>.</p>

<br>

<h2>Source</h2>

<h3>Original Owner:</h3>

<ul>
  <li>R. K. Bock</li>
  <li>Major Atmospheric Gamma Imaging Cherenkov Telescope project (MAGIC)</li>
  <li>http://wwwmagic.mppmu.mpg.de</li>
  <li>rkb '@' mail.cern.ch</li>
</ul>

<h3>Donor</h3>

<ul>
  <li>P. Savicky</li>
  <li>Institute of Computer Science, AS of CR</li>
  <li>Czech Republic</li>
  <li>savicky '@' cs.cas.cz</li>
</ul>

<br>

<h2>Table of Contents</h2>

<div class="alert alert-block alert-info" style="margin-top: 20px">
<li><a href="https://#import">Import Libraries</a></li>
<li><a href="https://#data_preparation">Dataset Preparation</a></li>
<li><a href="https://#compile_fit">Build and Fit the Model</a></li>
<li><a href="https://#analize_model">Analize the Model</a></li>
<li><a href="https://#revelant_papers">Relevant Papers</a></li>

<br>

<p></p>
Estimated Time Needed: <strong>25 min</strong>
</div>

<br>
<h2 align=center id="import">Import Libraries</h2>
<p>The following are the libraries we are going to use for this lab:</p>
"""

try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools

import keras

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

"""<br>
<h2 align=center id="data_preparation">Dataset Preparation (Data Preprocessing)</h2>

<p>Let's build some necessary functions for visualisation and pre-processing.</p>
"""

# Functions taken from 'Custom Models, Layers, and Loss Functions with TensorFlow' course.
# https://www.coursera.org/learn/custom-models-layers-loss-functions-with-tensorflow

def format_output(data):
    y1 = data.pop('le_class')
    y1 = np.array(y1)
    return y1

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

def plot_metrics(metric_name, title, ylim=5):
    plt.figure(figsize = (20, 10))
    sns.set_style('whitegrid')
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)
    plt.legend(['train', 'validation'])
    plt.show()

df = pd.read_csv('/content/magic04.data', sep=",", skiprows=1, header=None)
df

"""<p>Now, we can upload our data named <code>magic04.data</code>. It doesn't have named columns. We need to spesify them after importing our data into <i>dataframe</i>. Let's take a look at attribute information from <a href='https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope'>archive.ics.uci.edu</a> site.</p>

<h3>Attribute Information</h3>

<ol>
  <li>fLength: continuous (major axis of ellipse as <i>mm</i>)</li>
  <li>fWidth: continuous (minor axis of ellipse as <i>mm</i>)</li>
  <li>fSize: continuous (10-log of sum of content of all pixels <i>in photo</i>)</li>
  <li>fConc: continuous (ratio of sum of two highest pixels over fSize as <i>ratio</i>)</li>
  <li>fConc1: continuous (ratio of highest pixel over fSize as <i>ratio</i>)</li>
  <li>fAsym: continuous (distance from highest pixel to center, projected onto major axis as <i>mm</i>)</li>
  <li>fM3Long: continuous (3rd root of third moment along major axis as <i>mm</i>) </li>
  <li>fM3Trans: continuous (3rd root of third moment along minor axis as <i>mm</i>)</li>
  <li>fAlpha: continuous (angle of major axis with vector to origin as <i>deg</i>)</li>
  <li>fDist: continuous (distance from origin to center of ellipse as <i>mm</i>)</li>
  <li>class: g,h (as gamma (<i>signal</i>) and hadron (<i>background</i>))</li>
</ol>
"""

df.columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
df

print("Number of NaN values: {}.".format(df.isnull().sum().sum()))

print("Number of duplicated rows: {}.".format(df.duplicated().sum()))

"""<p>We have 115 duplicated rows. We need to drop them because we want the program to learn, not memorize.</p>"""

dp = df[df.duplicated(keep=False)]
df.drop_duplicates(inplace= True)
print("Number of duplicated rows: {}.".format(df.duplicated().sum()))

le = LabelEncoder()
df["le_class"] = le.fit_transform(df['class'])

"""<p>In a real-world scenario, you should shuffle the data.</p>"""

df = df.iloc[np.random.permutation(len(df))]

df.head()

df.drop('class', axis=1, inplace=True)

df.head()

df.describe().T

df.info()

df['le_class'].value_counts()

plt.figure(figsize = (20, 10))
sns.set_style('whitegrid')
sns.histplot(data=df['le_class'], kde=True)
plt.title("Frequency of Classes")
plt.xlabel("Classes")
plt.ylabel("Frequency")

plt.figure(figsize = (20, 20))
sns.heatmap(df.corr(), annot=True)

plt.figure(figsize = (20, 20))
sns.pairplot(df)

for i in range(10):
  label = df.columns[i]
  plt.figure(figsize = (20, 7))
  plt.hist(df[df['le_class']==0][label], color='purple', label="Gamma", 
           alpha=0.7, density=True, bins=15) # Gamma
  plt.hist(df[df['le_class']==1][label], color='blue', label="Hadron", 
           alpha=0.7, density=True, bins=15) # Hadron
  plt.title(label)
  plt.ylabel("Distribution")
  plt.xlabel(label)
  plt.legend()
  plt.show()

"""<p>Split the data into train and test with 80 train / 20 test:</p>"""

train, test = train_test_split(df, test_size=0.2)
train_stats = train.describe()

train_stats.pop('le_class')
train_stats = train_stats.transpose()
train_stats

train_Y = format_output(train)
test_Y = format_output(test)

len(train_Y)

len(test_Y)

"""<h3>Test Train Split</h3>

<p>Creating train and test dataset Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set. This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the model. Therefore, it gives us a better understanding of how well our model generalizes on new data.

We know the outcome of each data point in the testing dataset, making it great to test with! Since this data has not been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, it is truly an out-of-sample testing. Let's split our dataset into train and test sets. Around 80% of the entire dataset will be used for training and 20% for testing.</p>
"""

train, test = train_test_split(df, test_size=0.2, random_state = 1)

train, val = train_test_split(train, test_size=0.2, random_state = 1)

train_stats = train.describe().T
train_stats

train_stats = train.describe()
train_stats.pop('le_class')
train_stats = train_stats.transpose()

train_stats

"""<p>Now, we need to seperate our data as dependent and independent variables. <code>le_class</code>column is our independent variable. Except <code>le_class</code> columns are dependent variables.</p>

"""

# Function taken from 'Custom Models, Layers, and Loss Functions with TensorFlow' course.
# https://www.coursera.org/learn/custom-models-layers-loss-functions-with-tensorflow

def format_output(data):
    target = data.pop('le_class')
    target = np.array(target)
    return target

"""<p>Now, we can pop independent variables from <code>test</code>, <code>train</code> and <code>val</code> dataset."""

train_Y = format_output(train)

val_Y = format_output(val)

test_Y = format_output(test)

"""<p>For an instance, let's take a look at <code>train</code> dataframe:</p>"""

train.head()

"""<p>Now, we need to normalize the our dataset with the following formula.</p>

$$x_{norm} = \frac{x - \mu}{\sigma}$$
"""

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

norm_train_X = norm(train)
norm_val_X = norm(val) 
norm_test_X = norm(test)

"""<p>Like we did before, let's take a look at our <code>train</code> data again."""

norm_train_X.head()

"""<br>
<h2 align=center id="build_fit_model">Build and Fit the Model</h2>
"""

LR = 0.0001 #@param {type:"number"}
EPOCHS = 60 #@param {type:"number"}
INDEPENDENT_VARIABLES = len(train.columns)

"""<p>You can use the weights builded by myself. For that, please uncomment the following code line:</p>"""

#model = keras.models.load_model('/content/best_model.h5')

def base_model(inputs):
    
  x = Dense(units='128', activation='relu', name='base_dense_1')(inputs)
  x = Dropout(0.1, name="base_dropout_1")(x)
  x = Dense(units='128', activation='relu', name='base_dense_2')(x)
  x = Dropout(0.1, name="base_dropout_2")(x)
  x = Dense(units='64', activation='relu', name='base_dense_3')(x)
  x = Dropout(0.1, name="base_dropout_3")(x)
  x = Dense(units='32', activation='sigmoid', name='base_dense_4')(x)
  
  return x

def final_model(inputs):
    
  x = base_model(inputs)

  class_type = Dense(units='1', activation='sigmoid', name='class_type')(x)
  model = Model(inputs=inputs, outputs=class_type)

  print(model.summary())
  return model

inputs = tf.keras.layers.Input(shape=(INDEPENDENT_VARIABLES,))
rms = tf.keras.optimizers.RMSprop(lr=LR)
model = final_model(inputs)

from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

model.compile(optimizer=rms, 
            loss = {'class_type' : 'binary_crossentropy'},
            metrics = {'class_type' : 'accuracy'}
            )

checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

history = model.fit(norm_train_X, train_Y,
                    epochs = EPOCHS, validation_data=(norm_val_X, val_Y), 
                    callbacks=[callbacks_list])

"""<br>
<h2 align=center id="analize_model">Analize the Model</h2>
"""

def plot_metrics(metric_name, title, ylim=5):
  plt.figure(figsize = (20, 10))
  plt.title(title)
  plt.ylim(0,ylim)
  plt.plot(history.history[metric_name],color='purple',label=metric_name)
  plt.plot(history.history['val_' + metric_name],color='blue',label='val_' + metric_name)
  plt.legend(['train', 'validation'])
  plt.ylabel('Loss')
  plt.xlabel('Epoch')

# Function taken from 'Custom Models, Layers, and Loss Functions with TensorFlow' course.
# https://www.coursera.org/learn/custom-models-layers-loss-functions-with-tensorflow

def plot_confusion_matrix(y_true, y_pred, title='', labels=[0, 1]):
  cm = confusion_matrix(y_true, y_pred)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(cm)
  plt.title('Confusion Matrix of the Classifier')
  fig.colorbar(cax)
  ax.set_xticklabels([''] + labels)
  ax.set_yticklabels([''] + labels)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  fmt = 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="black" if cm[i, j] > thresh else "white")
  plt.show()

predictions = model.predict(norm_test_X)

np.round(predictions)

test_Y

plot_confusion_matrix(test_Y, np.round(predictions), title='Class Type', labels = [0, 1])

plot_metrics('loss', 'High Energy Gamma Particles', ylim=0.75)

plot_metrics('accuracy', 'High Energy Gamma Particles', ylim=1)

"""<p>Now, we can build our ROC curve for our <code>best_model</code>.</p>"""

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(test_Y, np.round(predictions))
fpr, tpr, thresholds = roc_curve(test_Y, np.round(predictions))

plt.figure(figsize=(20, 20))
plt.plot(fpr, tpr, label='Artificial Neural Network (best_model) (area = %0.3f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Artificial Neural Network (best_model)')
plt.legend(loc='upper left')
#plt.savefig('Log_ROC_curve')
plt.show()

"""<br>
<h2 align=center id="revelant_papers">Relevant Papers</h2>

<ul>
  <li><b>Bock, R.K., Chilingarian, A., Gaug, M., Hakl, F., Hengstebeck, T., Jirina, M., Klaschka, J., Kotrc, E., Savicky, P., Towers, S., Vaicilius, A., Wittek W. (2004).</b> <i>Methods for multidimensional event classification: a case study using images from a Cherenkov gamma-ray telescope. Nucl.Instr.Meth. A, 516, pp. 511-528.</i></li>
  <br>
  <li><b>P. Savicky, E. Kotrc. Experimental Study of Leaf Confidences for Random Forest.</b> <i>Proceedings of COMPSTAT 2004, In: Computational Statistics. (Ed.: Antoch J.) - Heidelberg, Physica Verlag 2004, pp. 1767-1774.</i></li>
  <br>
  <li><b>J. Dvorak, P. Savicky. Softening Splits in Decision Trees Using Simulated Annealing.</b> <i>Proceedings of ICANNGA 2007, Warsaw, (Ed.: Beliczynski et. al), Part I, LNCS 4431, pp. 721-729. </i></li>
<ul>

<br>

<h1>Contact Me</h1>
<p>If you have something to say to me please contact me:</p>

<ul>
  <li>Twitter: <a href="https://twitter.com/Doguilmak">Doguilmak</a></li>
  <li>Mail address: doguilmak@gmail.com</li>
</ul>
"""

from datetime import datetime
print(f"Changes have been made to the project on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")