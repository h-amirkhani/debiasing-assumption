from matplotlib.colors import SymLogNorm
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import pickle
from transformers import AutoTokenizer
import os, json
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, entropy, kendalltau, pearsonr
from scipy.special import softmax
import torch
import matplotlib.pyplot as plt
import seaborn as sns


data_source = 'fever'
bias_type = 'part' # part or tiny
corr = 'co' # sp(earman) or co(sine) or pe(arson) or ke(ndaltau)

threshold_dic = {'mnli_tiny':0.386076, 'mnli_part':0.269097, 'fever_part':0.3976218, 'fever_tiny':0.6183813}  # obtained through experiments

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # Tiny has the same tokenizer

if data_source == 'fever':
  partial = 'claim'
  label_dic = {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2}
  with open('./data/fever/fever.dev.jsonl', 'r') as json_file:
      json_list = list(json_file)
  fever_val_data = [json.loads(json_str) for json_str in json_list]
  first_input = [r['evidence'] for r in fever_val_data]
  second_input = [r['claim'] for r in fever_val_data]
  val_label = [label_dic[r['gold_label']] for r in fever_val_data]
  with open('./data/fever/new_dev.jsonl', 'r') as json_file:
    json_list = list(json_file)
    new_val_data = [json.loads(json_str) for json_str in json_list]
    for i,l in enumerate([r['label'] for r in new_val_data]):
      if l == 'NOT ENOUGH INFO':
        first_input.append(new_val_data[i]['evidence'])
        second_input.append(new_val_data[i]['claim'])
        val_label.append(1)
else:
  partial = 'hyp'
  dataset = load_dataset('glue', 'mnli')
  val_label = dataset['validation_matched']['label']
  first_input = dataset['validation_matched']['premise']
  second_input = dataset['validation_matched']['hypothesis']

# Assume the omission vectors are obtained throuh omission.py and saved
with open(f'./omission/{data_source}-bert-base-uncased.pickle', 'rb') as handle:
    main = pickle.load(handle)
if bias_type == 'tiny':
  with open(f'./omission/{data_source}-bert-tiny.pickle', 'rb') as handle:
    bias = pickle.load(handle)
elif bias_type == 'part':
  with open(f'./omission/{data_source}-{partial}-bert-base-uncased.pickle', 'rb') as handle:
    bias = pickle.load(handle)

main_dic = defaultdict(list)
bias_dic = defaultdict(list)
for m in main:
  main_dic[m[0]].append(m[1:])
for b in bias:
  bias_dic[b[0]].append(b[1:])
  
res = {}
for k,v in tqdm(main_dic.items()):
  L1 = []    
  found = False
  for l in v:
    if found or bias_type=='tiny':
        L1.append(l[1][val_label[k]])
    elif l[0] == '[SEP]':
      found = True

  L2 = [l[1][val_label[k]] for l in bias_dic[k]]
  if corr == 'co':
    res[k] = [val_label[k], 1 - cosine(L1,L2)] 
  elif corr == 'sp':
    res[k] = [val_label[k], spearmanr(L1,L2).correlation]
  elif corr == 'pe':
    res[k] = [val_label[k], pearsonr(L1, L2).correlation]
  elif corr == 'ke':
    res[k] = [val_label[k], kendalltau(L1, L2).correlation]

# Assume model predictions for validation data are obtained through training.py and saved
if bias_type == 'tiny':
  path = 'bert-tiny'
elif bias_type == 'part':
  path = f'{partial}-bert-base-uncased'
with open(f'./predictions/{data_source}-{path}.pickle', 'rb') as handle:
  pred = pickle.load(handle)
with open(f'./predictions/{data_source}-bert-base-uncased.pickle', 'rb') as handle:
  main_pred = pickle.load(handle)

print(f"Acc of val prediction: {np.mean(np.argmax(pred.predictions, axis=1) == pred.label_ids)*100}")

# statistics
print(f"Total# (main correct): {len(main_dic)}")  

L_easy = []
easy_pos = []
L_hard = []
for k,v in res.items():
  if np.argmax(pred.predictions[k]) == pred.label_ids[k]:
    L_easy.append(v[1])
    easy_pos.append(k)
  else:
    L_hard.append(v[1])
print(f"Easy#: {len(L_easy)} ({len(L_easy)/len(main_dic)*100}%)")
print(f"Hard#: {len(L_hard)} ({len(L_hard)/len(main_dic)*100}%)")
    
# overall correlation
L = []
for _,v in res.items():
  L.append(v[1])
print(f"Main-Biased corr (Total): {np.mean(L)}")
print(f"Main-Biased corr (Easy): {np.mean(L_easy)}")
print(f"Main-Biased corr (Hard): {np.mean(L_hard)}")

neg_easy = (np.array(L_easy) < threshold_dic[f"{data_source}_{bias_type}"]).sum()
print(f"Negative Correlation (Easy): {neg_easy}, {neg_easy / len(L_easy) * 100}")

easy_lab_neg = []
for i,v in enumerate(L_easy):
  if v<threshold_dic[f"{data_source}_{bias_type}"]:
    easy_lab_neg.append(val_label[easy_pos[i]])
easy_lab_neg = np.array(easy_lab_neg)
print(f"----Support 0: {np.sum(easy_lab_neg == 0)/ neg_easy*100}") 
print(f"----Support 1: {np.sum(easy_lab_neg == 1)/ neg_easy*100}") 
print(f"----Support 2: {np.sum(easy_lab_neg == 2)/ neg_easy*100}") 

easy_lab = []
for i in easy_pos:
  easy_lab.append(val_label[i])
easy_lab = np.array(easy_lab)
print('Easy distribution')
print(f"----Support 0: {np.sum(easy_lab == 0)/ len(easy_lab)*100}") 
print(f"----Support 1: {np.sum(easy_lab == 1)/ len(easy_lab)*100}") 
print(f"----Support 2: {np.sum(easy_lab == 2)/ len(easy_lab)*100}") 

# total distribution (correctly classified)
tot_lab = []
for k,v in res.items():
  tot_lab.append(val_label[k])
tot_lab = np.array(tot_lab)
print('All distribution (Correct)')
print(f"----Support 0: {np.sum(tot_lab == 0)/ len(tot_lab)*100}") 
print(f"----Support 1: {np.sum(tot_lab == 1)/ len(tot_lab)*100}") 
print(f"----Support 2: {np.sum(tot_lab == 2)/ len(tot_lab)*100}") 

# total distribution (ALL)
tot_lab = np.array(val_label)
print('All distribution')
print(f"----Support 0: {np.sum(tot_lab == 0)/ len(tot_lab)*100}") 
print(f"----Support 1: {np.sum(tot_lab == 1)/ len(tot_lab)*100}") 
print(f"----Support 2: {np.sum(tot_lab == 2)/ len(tot_lab)*100}") 

# confidences
main_correct = []
main_wrong = []
biased_correct = []
biased_wrong = []
for i in range(len(pred.predictions)):
  p_main = max(softmax(main_pred.predictions[i]))
  p_biased = max(softmax(pred.predictions[i]))
  if i in res:
    main_correct.append(p_main)
  else:
    main_wrong.append(p_main)
  if i in easy_pos:
    biased_correct.append(p_biased)
  else:
    biased_wrong.append(p_biased)
print('Confidence')
print(f"----Main Correct: {np.mean(main_correct)}")
print(f"----Main Wrong: {np.mean(main_wrong)}")
print(f"----Biase Correct: {np.mean(biased_correct)}")
print(f"----Biase Wrong: {np.mean(biased_wrong)}")


# Compute correlation between different corr measures
if LEasy not in locals():
  LEasy = dict()
LEasy[corr] = L_easy  

if ('co' in LEasy) and ('sp' in LEasy):
  print(f"Correlation between cosine and spearman (spearman): {spearmanr(LEasy['co'], LEasy['sp']).correlation}, (cosine): {1 - cosine(LEasy['co'], LEasy['sp'])}")

prob = []
for ind in easy_pos:
  prob.append(softmax(pred.predictions[ind])[val_label[ind]])
plt.scatter(L_easy, prob, marker='.')


# plot correlation distribution
sns.set(font_scale=2)
sns.set_style("whitegrid", {'axes.grid' : False})
ax = sns.histplot(L_easy, palette="light:m_r", edgecolor="0", linewidth=.5, bins=20)
if bias_type == 'part':
  bias_label = 'Partial'
else:
  bias_label = 'Tiny'
ax.set(title=f"{data_source.upper()}-{bias_label}")
if data_source == 'fever':
  ax.set(xlabel='Cosine')
if bias_type == 'part':
  ax.set(ylabel='Count')
else:
  ax.set(ylabel='')
ax.axvline(np.mean(L_easy), color='b', linestyle='-', linewidth=3)
ax.axvline(threshold_dic[f"{data_source}_{bias_type}"], color='r', linestyle='--', linewidth=3)
if data_source=='mnli' and bias_type=='part':
  ax.legend(['Mean', 'Threshold'])
ax.get_figure().savefig(f"{data_source}-{bias_type}-hist.pdf", bbox_inches='tight')

# plot distributions
fever_classes = ['Support', 'NEI', 'Refute']
mnli_classes = ['Entail', 'Neutral', 'Contradict']
dist_dic = {'MNLI-Tiny_easy':[40.9, 28.57, 30.53], 'MNLI-Tiny_dif':[24.53, 43.69, 31.78], 'MNLI-Tiny_all':[35.45, 31.82, 32.74],
            'MNLI-Partial_easy':[30.66, 33.27, 36.07], 'MNLI-Partial_dif':[53.80, 25.71, 20.49], 'MNLI-Partial_all':[35.45, 31.82, 32.74],
            'FEVER-Tiny_easy':[50.86, 17.21, 31.93], 'FEVER-Tiny_dif':[59.49, 25.19, 15.37], 'FEVER-Tiny_all':[39.92, 16.67, 43.41],
            'FEVER-Partial_easy':[47.34, 13.32, 39.34], 'FEVER-Partial_dif':[64.79, 20.30, 14.92], 'FEVER-Partial_all':[39.92, 16.67, 43.41]}  # Obtained throgh experiments
for setting in ['MNLI-Partial', 'MNLI-Tiny', 'FEVER-Tiny', 'FEVER-Partial']:
  key = setting + '_easy'
  easy_dist = dist_dic[key]
  key = setting + '_dif'
  dif_dist = dist_dic[key]
  key = setting + '_all'
  all_dist = dist_dic[key]

  assert np.abs(100-np.sum(easy_dist))<1
  assert np.abs(100-np.sum(dif_dist))<1
  assert np.abs(100-np.sum(all_dist))<1

  if setting.startswith('MNLI'):
    x = mnli_classes
  else:
    x = fever_classes

  sns.set(font_scale=2)
  sns.set_style("whitegrid", {'axes.grid' : False})
  df = pd.DataFrame(zip(x*3, all_dist+easy_dist+dif_dist, ['Val']*3+['Easy']*3+['Different']*3), columns=["Class", "distribution", "Instances"])
  ax = sns.barplot(x="Class", y="distribution", hue='Instances', data=df)
  ax.set(title=setting)
  ax.set(xlabel='')
  if setting != 'MNLI-Partial':
    plt.legend([],[], frameon=False)
  else:
    ax.legend(fontsize=15, loc='upper right').set_title('')
    ax.set(ylim=(0,60))
    
  if setting.endswith('Tiny'):
    ax.set(ylabel='')
  ax.get_figure().savefig(f'{setting}-distribution.pdf', bbox_inches='tight')
  plt.show()

# Qualitiative
ind = 18321 # index of one example
sm = softmax(pred.predictions[ind])
print(f"Bias model confidence towards true label: {sm[val_label[ind]]}")

sns.set(font_scale=2)
sns.set_style("whitegrid", {'axes.grid' : False})

L1 = [l[1][val_label[ind]] for l in main_dic[ind]] 
L2 = [l[1][val_label[ind]] for l in bias_dic[ind]] 

input_ids = tokenizer(first_input[ind], second_input[ind], truncation=True, padding=True)['input_ids']
input_ids = torch.tensor([input_ids], dtype=torch.long)
T1 = tokenizer.convert_ids_to_tokens(input_ids.view(-1).cpu().numpy())
if bias_type == 'part':
  input_ids = tokenizer(second_input[ind], truncation=True, padding=True)['input_ids']
  input_ids = torch.tensor([input_ids], dtype=torch.long)
T2 = tokenizer.convert_ids_to_tokens(input_ids.view(-1).cpu().numpy())

df = pd.DataFrame()
df['tokens'] = T1[1:-1]
df['Main'] = L1
df['Biased'] = [0]*(len(L1)-len(L2)) + L2
sns.set(font_scale=3.5)
for col,lab in zip(['Biased', 'Main'], [False, T1[1:-1]]):
  plt.figure(figsize=(40,2))
  dat = df[[col.strip()]].to_numpy().T
  ax = sns.heatmap(dat, square=True,
            xticklabels=lab, yticklabels=[col], cmap='Blues', linewidths=3, annot=True, fmt='0.1f',
            annot_kws={"fontsize":30, "color":'r'}, cbar=False, norm = SymLogNorm(1.5))
  plt.yticks(rotation=0)
  plt.xticks(rotation=60)
  plt.show()
  ax.get_figure().savefig(f'{col.strip()}.pdf', bbox_inches='tight')


### Manual labeling ###
# Save samples for manual labeling
titles = ['MAIN', 'BIASED']
def save_pic(ind): 
  L1 = [l[1][val_label[ind]] for l in main_dic[ind]] 
  L2 = [l[1][val_label[ind]] for l in bias_dic[ind]] 

  if bias_type == 'part':
    input_ids = tokenizer(second_input[ind], truncation=True, padding=True)['input_ids']
    input_ids = torch.tensor([input_ids], dtype=torch.long)
  else:
    input_ids = tokenizer(first_input[ind], second_input[ind], truncation=True, padding=True)['input_ids']
    input_ids = torch.tensor([input_ids], dtype=torch.long)
  T = tokenizer.convert_ids_to_tokens(input_ids.view(-1).cpu().numpy())
  L1 = L1[-len(T)+2:]

  plt.figure(figsize=(21,8))
  for sub,(tokens, gradients) in enumerate(zip([T[1:-1], T[1:-1]], [L1, L2])):
    plt.subplot(2,1,sub+1)
    xvals = [ x + str(i) for i,x in enumerate(tokens)]
    plt.tick_params(axis='both', which='minor', labelsize=29)
    p = plt.bar(xvals, gradients, linewidth=1)
    p = plt.xticks(ticks=[i for i in range(len(tokens))], labels=tokens, fontsize=12,rotation=90) 
    plt.title(titles[sub])
  plt.subplots_adjust(hspace=1)
  if not os.path.isdir(f"{data_source}-{bias_type}"):
    os.mkdir(f"{data_source}-{bias_type}")
  plt.savefig(f"{data_source}-{bias_type}/{str(ind)}.jpg")
  plt.close()

n_samples = 250
n_bins = 10
samples_per_bin = round(n_samples/n_bins)
pick_samples = samples_per_bin

mn = min(L_easy)
mx = max(L_easy)
width = (mx-mn)/n_bins

arr = np.array(L_easy)
pick_pos = []

prev_start = False
for i in range(n_bins):
  if not prev_start:  # not enough samples in the previous bin
    start = mn + width * i
  end = mn + width * (i+1)
  found_list = np.where(np.logical_and(arr>=start,arr<end))[0]
  if len(found_list) < pick_samples:
    prev_start = True
    pick_samples += samples_per_bin
  else:
    np.random.shuffle(found_list)
    for f in found_list[:pick_samples]:
      pick_pos.append(easy_pos[f])
    prev_start = False
    pick_samples = samples_per_bin

for p in tqdm(pick_pos, leave=True, position=0):
  save_pic(p)

# check, shuffle, and save the ids
temp_arr = np.array(pick_pos)
np.random.shuffle(temp_arr)
temp_cosine = []
for l in temp_arr:
  temp_cosine.append(L_easy[np.where(easy_pos==l)[0][0]])
plt.hist(temp_cosine)
labelers = ['HT']*40 + ['T']*105 + ['H']*105  # H stands for Hossein and T stands for Taher (the authors)
df = pd.DataFrame(np.c_[temp_arr, labelers], columns=['id','labeler'])
df.to_excel(f'{data_source}-{bias_type}.xlsx', index=False)

# After manual labeling by labelers
labeler2 = pd.read_excel(f"{data_source}-{bias_type}-labeler2.xlsx")
labeler2_id = list(labeler2['id'])
labeler2_labelers = list(labeler2['labeler'])
labeler2_label = list(labeler2['label'])
labeler2_doubt = list(labeler2['doubt'])
labeler2_dic = {}
for i in range(len(labeler2)):
  if not np.isnan(labeler2_label[i]):
    labeler2_dic[labeler2_id[i]] = (labeler2_labelers[i], labeler2_label[i], labeler2_doubt[i])

labeler1 = pd.read_excel(f"{data_source}-{bias_type}-labeler1.xlsx")
labeler1_id = list(labeler1['id'])
labeler1_labelers = list(labeler1['labeler'])
labeler1_label = list(labeler1['label'])
labeler1_doubt = list(labeler1['doubt'])
labeler1_dic = {}
for i in range(len(labeler1)):
  if not np.isnan(labeler1_label[i]):
    labeler1_dic[labeler1_id[i]] = (labeler1_labelers[i], labeler1_label[i], labeler1_doubt[i])

# agreement between labelers
disagree_list = []
cosine_common = []
agree_common = []

for k,v in labeler1_dic.items():
  if v[0] == 'HT' and (not np.isnan(v[1])) and labeler2_dic[k][0]=='HT' and (not np.isnan(labeler2_dic[k][1])):
    cosine_common.append(L_easy[np.where(np.array(easy_pos)==k)[0][0]])
    if v[1] == labeler2_dic[k][1]:
      agree_common.append(1)
    else:
      agree_common.append(0)
      disagree_list.append(k)
print(f'Labelers: agree ({np.sum(agree_common)}), Disagree ({len(cosine_common)-np.sum(agree_common)})')

# agreement between labelers and classifier
L1 = []
L2 = []
for k,v in labeler2_dic.items():
  if k not in disagree_list:
    L1.append(v[1])
    ind = np.where(np.array(easy_pos) == k)[0][0]
    L2.append(L_easy[ind])

for k,v in labeler1_dic.items():
  if k not in disagree_list:
    L1.append(v[1])
    ind = np.where(np.array(easy_pos) == k)[0][0]
    L2.append(L_easy[ind])

print(f"Spearman: {spearmanr(L1,L2)}")
print(f"AUC: {roc_auc_score(L1, L2)}")

# roc curves
fpr, tpr, thresholds = roc_curve(L1, L2)
plt.plot([0,1], [0,1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# optimal threshold
thresholds = []
precision = []
recall = []
fmeasure = []
labels = np.array([1-l for l in L1])
corrs = np.array(L2)
for th in corrs:
  thresholds.append(th)
  arr = corrs <= th
  precision.append(np.sum(np.bitwise_and(arr, labels==1))/np.sum(arr))
  recall.append(np.sum(np.bitwise_and(arr, labels==1))/np.sum(labels))
  fmeasure.append((2*precision[-1]*recall[-1]) / (precision[-1] + recall[-1]))
ix = np.argmax(fmeasure)
print(thresholds[ix], fmeasure[ix])
print(f"Ratio of samples not following Bias model (F1): {np.sum(np.array(L_easy)<thresholds[ix])} - {np.mean(np.array(L_easy)<thresholds[ix])}")
