#Read list of images
#compute features and show
#visualize

# datos https://www.dropbox.com/scl/fi/d8r23u89vlaym74i25sto/personas.zip?rlkey=bwo587id3j4sb0pgpx8mhuctq&st=808ivdwn&dl=0
import os
import PIL 
import PIL.Image
import numpy as np
from insightface.app import FaceAnalysis

datapath = '/hd_data/personas'
ffaces = os.path.join(datapath, 'faces.txt')

# Initialize face analysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=0)  # ctx_id=-1 for CPU, 0 for GPU

def get_face_embedding(image_path):
    """Extract face embedding from an image"""
    img = PIL.Image.open(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    faces = app.get(np.array(img))
    print(faces)
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    
    return faces[0].embedding

# def compare_faces(emb1, emb2, threshold=0.65): # Adjust this threshold according to your usecase.
#     """Compare two embeddings using cosine similarity"""
#     similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
#     return similarity, similarity > threshold

# Paths to your Indian face images
embs = []
names = []
compute = False 
with open(ffaces)  as f: 
    for line in f : 
        iname, icl = line.split()
        iname = os.path.join(datapath, iname.strip())
        print(iname)
        names.append(icl)
        if compute :
            emb = get_face_embedding(iname)
            if len(embs) == 0 :
                embs = emb
            else :
                embs = np.vstack([embs, emb])

# print(embs.shape)
# np.save("face_embs.npy", embs)
embs = np.load("face_embs.npy") 
#similarity search
embs_norm = embs / np.linalg.norm(embs, ord = 2, axis = 1,  keepdims = True)
sim = embs_norm @ np.transpose(embs_norm)
idx_sort =  np.argsort(-sim, axis = 1)
print(idx_sort)
for  idx, row in enumerate(idx_sort) :    
    result = [(names[i], sim[idx, i]) for i in row[:5] ]
    print(result)

#view
labels = set(names)
name_to_id = {s: i for i, s in enumerate(labels)}
intlabels = range(0, len(labels))
intnames = np.array([name_to_id[name] for name in names])
import umap
import seaborn as sns
import matplotlib.pyplot as plt
color_palette = sns.color_palette( n_colors=len(labels))
color_map = dict(zip(intlabels, color_palette))
reducer = umap.UMAP(n_components = 2, min_dist = 0.1, n_neighbors = 2)
embedding = reducer.fit_transform(embs)
print(embedding.shape)
print(names)
print(labels)
for label in intlabels :            
    print(label)
    ids = np.where(intnames == label)[0]    
    print(ids)
    x = embedding[ids, 0]
    y = embedding[ids, 1]
    plt.scatter(
        x,
        y, color = color_map[label], label = label)

plt.legend(labels)
plt.title('UMAP projection of the MNIST dataset', fontsize=24)
plt.show()


        