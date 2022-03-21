import numpy as np
from keras import models
from keras import layers
import pandas as pd
import csv

kmers=pd.read_csv('/Users/a1667917/Documents/DeepLearning/Uniprot/kmer_counts.csv', sep=',',header=None)
go=pd.read_csv('/Users/a1667917/Documents/DeepLearning/Uniprot/total_go.csv', sep=',',header=None)


#go_names=pd.read_csv('/Users/a1667917/Documents/DeepLearning/Uniprot/go_names.csv', sep=',',header=None)
with open('/Users/a1667917/Documents/DeepLearning/Uniprot/go_names.csv', newline='') as f:
    reader = csv.reader(f)
    go_names = list(reader)

with open('/Users/a1667917/Documents/DeepLearning/Uniprot/genes.csv', newline='') as f:
    reader = csv.reader(f)
    genes = list(reader)



kmers_bio=pd.read_csv('/Users/a1667917/Documents/DeepLearning/Uniprot/kmer_counts_biofilm.csv', sep=',',header=None)
kmers_bio = np.asarray(kmers_bio)


combo = pd.concat([kmers.reset_index(drop=True), go], axis=1)

shuffled = combo.sample(frac=1)

go_terms = 147

go_df = shuffled.iloc[: , -go_terms:]
kmers = shuffled.iloc[: ,:-go_terms]

# get test genes

test_genes = [0] * 1000

indices =  list(kmers.iloc[:1000,:].index)

for i in range(1000):
    test_genes[i] = genes[indices[i]]





#### do for all go terms 

kmers = np.asarray(kmers)
go_df = np.asarray(go_df)

kmers_test = kmers[:1000]
kmers_val = kmers[1000:3000]
partial_kmers_train = kmers[3000:]
go_test_tot = go_df[:1000]
go_val_tot = go_df[1000:3000]
partial_go_train_tot = go_df[3000:]

predictions =  np.zeros((1000, 1))
results = np.zeros((2,1))

bio_preds = np.zeros((4, 1))

for i in range(go_terms):
    go_test =  go_test_tot[:,i]
    go_val = go_val_tot[:,i]
    partial_go_train = partial_go_train_tot[:,i]
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(400,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])
    model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc'])
    history = model.fit(partial_kmers_train,
    partial_go_train,
    epochs=5,
    batch_size=128,
    validation_data=(kmers_val, go_val))
    nam = str(go_names[i]) 
    nam = nam.replace("[", "")
    nam = nam.replace("]", "")
    nam = nam.replace(":", "_")
    nam = nam.replace("\'", "")
    model.save( 'models/' + nam )
    #reconstructed_model = models.load_model(nam)
    b = model.predict(kmers_test)
    r = model.evaluate(kmers_test, go_test)
    b = np.rint(b)
    p = model.predict(kmers_bio)
    p = np.rint(p)
    predictions = np.column_stack((predictions,b))
    results = np.column_stack((results,r))
    bio_preds = np.column_stack((bio_preds,p))

# round to nearest int

predictions = np.rint(predictions)
predictions = predictions[:,1:]

predictions_with_gene = np.column_stack((test_genes,predictions))

pdf = pd.DataFrame(predictions_with_gene)

import itertools

flat_go_names = itertools.chain(*go_names)
flat_go_names = list(flat_go_names) 

flat_go_names.insert(0,'gene')


bio_preds_pd = pd.DataFrame(bio_preds)

pdf.columns = flat_go_names
bio_preds_pd.columns = flat_go_names

pd.DataFrame(pdf).to_csv("predictions.csv", index=False)

pd.DataFrame(bio_preds_pd).to_csv("bio_predictions.csv", index=False)




p = model.predict(kmers_bio)







model = models.load_model('models/'+'GO_0045892')


# plotting

import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

# plot loss




plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot accuracy 

plt.clf()

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

a = np.zeros((1000, 1))
b = model.predict(kmers_test)
np.column_stack((a,b))



results = model.evaluate(kmers_test, go_test)