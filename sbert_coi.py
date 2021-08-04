"""
Inputs a csv, reads the portion of each line after datetime.
These data are mapped to embeddings and then s-bert agglomerative clustering with a threshold is applied.
"""
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

coilist = []

#read data from csv
with open('statecoi.csv','r') as f: #insert csv name at statecoi.csv
    lines = f.readlines()
    lines.pop(0)                    #pops header
    for line in lines:
        if line[0 : 1].isdigit():
            coilist.append(line)
        else:
            x= len(coilist)
            coilist[x-1] = coilist[x-1] + ' ' + line
f.close()

corpus = []

#get everything after datetime
for coi in coilist:
    count = 10
    for i in range(0, len(coi)-1):
        if coi[i] == ',' and count == 0:
            corpus.append(coi[i:])
            break
        elif coi[i] == ',':
            count = count-1
        else:
            continue


corpus_embeddings = embedder.encode(corpus)

# Normalize the embeddings to unit length
corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

# Perform kmean clustering – change distance_threshold to change number of clusters
clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=2.0)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in clustered_sentences.items():
    print("Cluster ", i+1)
    print(cluster)
    for coi in cluster:
        words = coi.split(' ')
    print("")