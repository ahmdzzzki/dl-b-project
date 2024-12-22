#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# In[6]:


binetflow_file1 = "/home/ahmadzaki/Documents/dl-b-project/2/capture20110811.binetflow"


df1 = pd.read_csv(binetflow_file1)
# df2 = pd.read_csv(binetflow_file2)
print(df1.head())


# In[7]:


df1.head()


# In[8]:


df1.info()


# In[9]:


df1.isnull().sum()


# In[10]:


# Mengisi nilai yang hilang
df1['Sport'].fillna(df1['Sport'].mode()[0], inplace=True)
df1['Dport'].fillna(df1['Dport'].mode()[0], inplace=True)
df1['State'].fillna(df1['State'].mode()[0], inplace=True)
df1['sTos'].fillna(df1['sTos'].median(), inplace=True)
df1['dTos'].fillna(df1['dTos'].median(), inplace=True)


# In[11]:


# Mengonversi kolom 'StartTime' ke tipe datetime
df1['StartTime'] = pd.to_datetime(df1['StartTime'])

# Mengonversi kolom kategorikal menjadi numerik (contoh: Sport, Dport, Proto, Dir)
df1['Sport'] = pd.to_numeric(df1['Sport'], errors='coerce')
df1['Dport'] = pd.to_numeric(df1['Dport'], errors='coerce')

# Menggunakan LabelEncoder untuk mengubah kolom kategorikal menjadi numerik

# Kolom yang akan di-encode
categorical_columns = ['Proto', 'Dir', 'State', 'Label']

# Melakukan encoding
encoder = LabelEncoder()
for col in categorical_columns:
    df1[col] = encoder.fit_transform(df1[col])


# In[12]:


# Ekstraksi fitur waktu
df1['Hour'] = df1['StartTime'].dt.hour
df1['Minute'] = df1['StartTime'].dt.minute
df1['Second'] = df1['StartTime'].dt.second


# In[13]:


df1.drop(columns=['StartTime'], inplace=True)


# In[14]:


# Menentukan fitur numerik yang akan dinormalisasi
numerical_features = ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes', 'sTos', 'dTos', 'Sport', 'Dport', 'Hour', 'Minute', 'Second']

# Inisialisasi Scaler
scaler = StandardScaler()

# Menormalkan data
df1[numerical_features] = scaler.fit_transform(df1[numerical_features])

# Cek hasil normalisasi
print(df1.head())


# In[15]:


df1['Proto'].value_counts()


# In[16]:


# Memisahkan fitur (X) dan label (y)
X = df1.drop(columns=['Label'])  # Semua kolom kecuali 'Label'
y = df1['Label']  # Kolom 'Label' sebagai target


# In[17]:


# Membagi data menjadi training dan testing (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cek dimensi data pelatihan dan pengujian
print(X_train.shape, X_test.shape)


# In[18]:


# Fungsi untuk membuat sequences (urutan)
def create_sequences(data, labels, sequence_length):
    sequences = []
    label_sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label_seq = labels[i:i + sequence_length]
        sequences.append(seq)
        label_sequences.append(label_seq)
    return np.array(sequences), np.array(label_sequences)

# Misalnya, kita ingin membuat sequence dengan panjang 50 timestep
sequence_length = 50
X_sequences, y_sequences = create_sequences(X_train.values, y_train.values, sequence_length)

# Menampilkan dimensi data setelah menjadi sequences
print(X_sequences.shape, y_sequences.shape)


# In[27]:


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detected:", gpus)
    try:
        # Mengatur penggunaan memori GPU secara dinamis
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU memory growth set.")
    except RuntimeError as e:
        print("Error:", e)
else:
    print("No GPU detected, using CPU instead.")


# In[21]:


# Menonaktifkan XLA JIT untuk menghindari error pada kompilasi
tf.config.optimizer.set_jit(False)


# In[22]:


from tensorflow.keras import mixed_precision

# Mengaktifkan mixed precision training untuk menghemat memori dan mempercepat pelatihan
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


# In[ ]:


# Model lebih sederhana tanpa LSTM
model = Sequential()

# Hanya Dense layer untuk debugging
model.add(Dense(64, activation='relu', input_shape=(X_sequences.shape[1], X_sequences.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Kompilasi model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Latih model
history = model.fit(X_sequences, y_sequences, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# In[ ]:


# Membuat model LSTM
model = Sequential()

# LSTM Layer
model.add(LSTM(units=64, return_sequences=False, input_shape=(X_sequences.shape[1], X_sequences.shape[2])))
model.add(Dropout(0.5))

# Dense Layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output Layer (Binary Classification)
model.add(Dense(1, activation='sigmoid'))

# Kompilasi Model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Ringkasan model
model.summary()


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

# Prediksi
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Evaluasi kinerja
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

