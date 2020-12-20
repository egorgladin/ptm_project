import artm
import pickle
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from functools import partial
from datetime import datetime
from time import time
from copy import deepcopy
from itertools import product
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_model(bv, n_topics=10):
    model = artm.ARTM(num_topics=n_topics, num_document_passes=3, dictionary=bv.dictionary,
                      class_ids={'@default_class': 1.0})
    model.scores.add(artm.PerplexityScore(name='perplexity', dictionary=bv.dictionary))
    model.scores.add(artm.TopTokensScore(name='top-tokens', num_tokens=10))
    model.scores.add(artm.SparsityPhiScore(name='sparsity'))

    start = time()
    model.fit_offline(bv, num_collection_passes=10)
    print(f"fitting PLSA took {time() - start}s")
    return model

  
def add_reg(model, bv, portion_back=0.1, tau_back=None, tau_sparse=None, tau_decor=None):
    if tau_back is not None:
        n_back = int(n_topics * portion_back)
        model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='smooth', tau=tau_back,
                                                               dictionary=bv.dictionary,
                                                               topic_names=model.topic_names[:n_back]))
    if tau_sparse is not None:
        model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse', tau=tau_sparse,
                                                               dictionary=bv.dictionary,
                                                               topic_names=model.topic_names[n_back:]))
    if tau_decor is not None:
        model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator', tau=tau_decor))

    start = time()
    model.fit_offline(bv, num_collection_passes=10)
    print(f"Tuning artm took {time() - start}s")
    return model

  
def classify(artm_model, bv_train, bv_val, cls, train_target, val_target):
    train_X = artm_model.transform(batch_vectorizer=bv_train).T.to_numpy()
    val_X = artm_model.transform(batch_vectorizer=bv_val).T.to_numpy()
    cls.fit(X=train_X, y=train_target)
    return f1_score(y_true=val_target, y_pred=cls.predict(X=val_X), average='micro')
  
  
with open('train_target.pickle', 'rb') as handle:
    train_target = pickle.load(handle)
    
with open('val_target.pickle', 'rb') as handle:
    val_target = pickle.load(handle)

train_vw_file = 'train_vw'
val_vw_file = 'val_vw'

bv_train = artm.BatchVectorizer(data_path=train_vw_file, data_format='vowpal_wabbit',
                                batch_size=10000, target_folder='batches/train')
bv_val = artm.BatchVectorizer(data_path=val_vw_file, data_format='vowpal_wabbit',
                              batch_size=10000, target_folder='batches/val')

N_topics = [10, 20]

taus_back = [1e2, 1e3]
portions_back = [0.1, 0.2, 0.3, 0.4]
taus_sparse = [None, -10, -1]
taus_decor = [None, 1e3, 1e4]

classifiers = ['LogReg']
def name_to_cls(name):
    if name == 'LogReg':
        return LogisticRegression(max_iter=50)
    elif name == 'SVM':
        return SVC()

n_topics = 10
portion_back = 0.2
grid = product(taus_back, taus_sparse, taus_decor)
model = get_model(bv_train, n_topics=n_topics)
for tau_back, tau_sparse, tau_decor in tqdm(grid):
    parameters = {'n_topics': n_topics, 'portion_back': portion_back, 'tau_back': tau_back,
                  'tau_sparse': tau_sparse, 'tau_decor': tau_decor}
    model_ = add_reg(deepcopy(model), bv_train, portion_back=portion_back, tau_back=tau_back,
                     tau_sparse=tau_sparse, tau_decor=tau_decor)
    for cls_name in classifiers:
        parameters['cls'] = cls_name
        f1 = classify(model_, bv_train, bv_val, name_to_cls(cls_name), train_target, val_target)
        top_tokens = model_.score_tracker['top-tokens'].last_tokens
        result = {'parameters': parameters,
                  'perplexity': model_.score_tracker["perplexity"].value,
                  'sparsity': model_.score_tracker["sparsity"].value,
                  'top-tokens': [top_tokens[topic_name] for topic_name in model_.topic_names],
                  'f1': f1}
        with open(f'{datetime.now().strftime("%H:%M:%S")}.pickle', 'wb') as handle:
            pickle.dump(result, handle)

