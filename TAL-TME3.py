
# coding: utf-8

## TAL - TME3 - Classification de documents

### Chargement du corpus movies1000 (lecture de fichiers)

# In[95]:



import nltk.corpus.reader as pt

def getDocs(path):
    rdr = pt.CategorizedPlaintextCorpusReader(path, r'.*\.txt', cat_pattern=r'(\w+)/*')
    docs = [[rdr.raw(fileids=[f]) for f in rdr.fileids(c) ] for c in rdr.categories()]
    return docs

def getAllDocs(docs):
    all_docs = docs[0] + docs[1]
    return all_docs

def getAllLabels(docs):
    len_sad = len(docs[0])
    len_happy = len(docs[1])
    all_labels = np.ones(len_sad + len_happy)
    all_labels[:len_happy] = 0
    return all_labels


### Création des stopwords

# In[127]:

from nltk import download

def nltkDownload():
    """ Vous n'aurez surement pas les stopwords, il vous faudra les télécharger.
    """
    download()


# In[126]:

from nltk.corpus import stopwords

def makeNltkStopWords(languages=['french', 'english', 'german', 'spanish']):
    stop_words = []
    for l in languages:
        for w in stopwords.words(l):
           stop_words.append(w.encode('utf-8')) #w.decode('utf-8') buggait... avec certains caractères
    return stop_words


### Vectorisation et normalisation

# In[107]:

import sklearn.feature_extraction.text as txt
    
def fromAllDocsToBow(all_docs, strip_accents=u'ascii', lowercase=True,                      preprocessor=None, stop_words=None, token_pattern=u"[\\w']+\\w\\b",                      analyzer=u'word', max_df=1.0, max_features=20000, vocabulary=None,                      binary=True, ngram_range=(1, 1), min_df=0.0025):
    """ Depuis un liste de documents, génère une matrice sparse contenant les occurences des mots.
        A chaque mot est associé un identifiant grace à une table de hashage.
    """
    vec_param = txt.CountVectorizer(all_docs)
#strip_accents, lowercase, preprocessor, stop_words, token_pattern, vocabulary, binary, ngram_range)#, min_df)
    # analyzer, max_df, max_features
    bow = vec_param.fit_transform(all_docs)
    bow = bow.tocsr() # pour pouvoir print
    return bow, vec_param

def normalizeBow(bow):
    """ La somme de toutes les occurences des mots devient égale à 1
    """
    transformer = txt.TfidfTransformer(use_idf=False, smooth_idf=False)
    bow_norm = transformer.fit_transform(bow)
    return bow_norm   


### Transformation inverse

# In[114]:

import scipy.sparse as sp

def fromArgsToWords(args, vec):
    """ A partir d'une liste d'arguments obtenus par l'extraction des coefficients de notre modèle
        et d'une fonction de vectorisation, rend une liste de mots.
    """
    nb = len(args)
    matrix = sp.coo_matrix((np.ones(nb), (np.zeros(nb),args)))
    words = vec.inverse_transform(matrix)
    return words

def fromBowToWords(bow, vec):
    """ A partir d'une matrice sparce, rend les mots associés aux identifiants générés par
        la fonction de hashage lors de la vectorisation.
    """
    bow_inv = vec.inverse_transform(bow)
    return bow_inv


### Construction d'un classifier

# In[100]:

import numpy as np
import sklearn.naive_bayes as nb
from sklearn import svm
from sklearn import linear_model as lin
from sklearn import cross_validation

def crossValidation(clf, bow, all_labels, cv=5):
    X = bow
    y = all_labels
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    np_scores = np.array(scores)
    mean = np_scores.mean()
    std = np_scores.std()
    return scores, mean, std 

def fit(clf, bow, all_labels):
    """ Indispensable pour obtenir les clf.coef_ utile à la descrimination des mots """
    X = bow
    y = all_labels
    clf.fit(X, y)
    return clf

def predict(clf, docs):
    return clf.predict(docs)


### Mots les plus discriminants

# In[115]:

def mostDescriminantWords(clf, vec, nb_words=100):
    """ Testé avec svm.LinearSVC() """
    args_sort = clf.coef_.reshape(clf.coef_.shape[1]).argsort()
    args_pos = args_sort[:nb_words]
    words_pos = fromArgsToWords(args_pos, vec)
    args_neg = args_sort[-nb_words:]
    words_neg = fromArgsToWords(args_neg, vec)
    return words_pos, words_neg


### Main()

# In[119]:

# Chargement du corpus movies1000
path = '/Users/Tamazy/Dropbox/_Docs/UPMC/TAL/TME3/movies1000'
docs = getDocs(path)
all_docs = getAllDocs(docs) # liste contenant l'ensemble des documents du corpus d'apprentissage
all_labels = getAllLabels(docs)

print "Le contenu du premier document :"
print all_docs[0][:100]

print "Le label associé :", all_labels[0]


# In[123]:

# Paramétrage
languages = ['french', 'english', 'german', 'spanish']
stop_words = makeNltkStopWords(languages) # si erreur executez nltkDownload()
analyzer = u'word' # {‘word’, ‘char’, ‘char_wb’}
ngram_range = (1, 1) # unigrammes
lowercase = True
token_pattern = u"[\\w']+\\w\\b" # 
max_df = 1.0 #default
min_df = 5. * 1./len(all_docs) # on enleve les mots qui apparaissent moins de 5 fois
max_features = 20000 # nombre de mots au total dans notre matrice sparse
binary = True # presence coding or counting
strip_accents = u'ascii' #  {‘ascii’, ‘unicode’, None}
preprocessor=None
vocabulary=None

# Vectorisation
bow, vec = fromAllDocsToBow(all_docs, strip_accents, lowercase, preprocessor,                             stop_words, token_pattern, analyzer, max_df, max_features,                             vocabulary, binary, ngram_range, min_df)

print "Mots vectorisés du second document :"
print bow[1]

# /!\ Le premier indice est toujours 0 si on print bow[i]
# /!\ alors que si on print bow le premier indice changera
# /!\ en fonction du document


# In[121]:

# Normalisation
bow = normalizeBow(bow)

print "Après normalisation :"
print bow[1]


# In[125]:

# Modèles
clf = svm.LinearSVC() # SVM
clf_nb = nb.MultinomialNB() # Naive Bayes
clf_rl = lin.LogisticRegression() # regression logistique

# Cross-Validation
scores, mean, std  = crossValidation(clf, bow, all_labels, cv=5)

print "Scores obtenus avec crossValidation :", scores
print "Moyenne :", mean
print "Ecart type :", std


# In[117]:

# Mots les plus discriminants
clf = fit(clf, bow, all_labels) # afin de pouvoir récupérer les coefficients du clf
words_pos, words_neg = mostDescriminantWords(clf, vec, nb_words=100)
print "Mots les plus discriminants"
print "bad] Pour décrire les mauvais films: ", words_pos
print "good] Pour décrire les bons films: ", words_neg


# In[47]:




# In[ ]:



