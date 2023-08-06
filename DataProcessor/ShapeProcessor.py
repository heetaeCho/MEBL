import numpy as np
import torch
import copy
import json
import string

from nltk.corpus import stopwords
from torch.utils.data import DataLoader
from Models.EmbeddingModel import EmbeddingModel
from DataProcessor.SourceCodeProcessor import readCode
from tqdm import tqdm
import pickle

ebm = EmbeddingModel()
keywords = json.load(open("./DataProcessor/keywords.json"))["keywords"]
punctuation = string.punctuation
try:
    stop_words = stopwords.words('english')
except:
    import nltk
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

def getDataLoader(bug_reports, test=False):
    text_helper_path = './Helper/AspectJ_text_embedding_helper_no_padding.pkl'
    code_helper_path = './Helper/AspectJ_code_embedding_helper_no_padding.pkl'
    text_embedding_helper, code_embedding_helper = _loadHelper(text_helper_path, code_helper_path)
        
    full_data = []
    for bug_report in tqdm(bug_reports, ncols=70, desc=" Now Embedding: ", leave=True):
        if not test:
            _sampleFalse(bug_report)
        
        if bug_report.getBugId() in text_embedding_helper.keys():
            nl = text_embedding_helper[bug_report.getBugId()]
        else:
            text = bug_report.getSummary() + bug_report.getDescription()
            text = text.lower()

            new_text = []
            for token in text.split():
                if token in stop_words:
                    continue
                new_text.append(token)
            text = ' '.join(new_text)

            nl = ebm.embedding('nl', text)
            # nl = nl.view(-1, 512, 768).detach().cpu().numpy()
            nl = nl.view(1, -1, 768).detach().cpu().numpy()
            text_embedding_helper[bug_report.getBugId()] = nl
        
        for file in bug_report.getNewFiles():
            if file in code_embedding_helper.keys():
                sc_nl, sc_pl = code_embedding_helper[file]
            else:
                sc_nl, sc_pl = _codeEmbedding(file)
                if sc_nl is None: continue
                sc_nl = sc_nl.view(1, -1, 768).detach().cpu().numpy()
                sc_pl = sc_pl.view(1, -1, 768).detach().cpu().numpy()
                code_embedding_helper[file] = (sc_nl, sc_pl)
            full_data.append( (nl, sc_nl, sc_pl, 1) )

        if test:
            false_files = bug_report.false_candidates
        else:
            false_files = bug_report.false_code_files

        if test:
            false_files = tqdm(false_files, desc="False Files", ncols=70)

        for file in false_files:
            if file in code_embedding_helper.keys():
                sc_nl, sc_pl = code_embedding_helper[file]
            else:
                sc_nl, sc_pl = _codeEmbedding(file)
                if sc_nl is None: continue
                sc_nl = sc_nl.view(1, -1, 768).detach().cpu().numpy()
                sc_pl = sc_pl.view(1, -1, 768).detach().cpu().numpy()
                code_embedding_helper[file] = (sc_nl, sc_pl)
            full_data.append( (nl, sc_nl, sc_pl, 0) )

    #         if ix == 3:
    #             break

    # temp_sc_nl = []
    # temp_sc_pl = []
    # for data in full_data:
    #     _, sc_nl, sc_pl, _ = data
    #     temp_sc_nl.append(sc_nl)
    #     temp_sc_pl.append(sc_pl)

    # print("=================================================\nsc_nl\n")
    # for sc_nl in temp_sc_nl:
    #     print(sc_nl)
    # print("=================================================\nsc_pl\n")
    # for sc_pl in temp_sc_pl:
    #     print(sc_pl)
    # exit()

    _saveHelper(text_embedding_helper, code_embedding_helper, text_helper_path, code_helper_path)

    # dataloader = DataLoader(full_data, batch_size=1, shuffle=True)
    # dataloader = DataLoader(full_data, batch_size=1, shuffle=False)
    dataloader = full_data
    return dataloader

def _loadHelper(text_helper_path, code_helper_path):
    try:
        with open(text_helper_path, 'rb') as fr:
            text_embedding_helper = pickle.load(fr)
        print("===== Load Helper =====")
    except:
        text_embedding_helper = {}
        print("===== Generate Helper =====")
    try:
        with open(code_helper_path, 'rb') as fr:
            code_embedding_helper = pickle.load(fr)
    except:
        code_embedding_helper = {}
    return text_embedding_helper, code_embedding_helper

def _saveHelper(text_embedding_helper, code_embedding_helper, text_helper_path, code_helper_path):
    with open(text_helper_path, 'wb') as fw:
        pickle.dump(text_embedding_helper, fw)
    with open(code_helper_path, 'wb') as fw:
        pickle.dump(code_embedding_helper, fw)

def _codeEmbedding(code_file):
    sc_nl_vecs = []
    sc_pl_vecs = []

    code_file, is_error = readCode(code_file)
    if is_error == True or len(code_file.getCodeChunks()) == 0:
        return None, None

    for code_chunk in code_file.getCodeChunks():
        if "Import" in code_chunk.getDeclaration() or "Package" in code_chunk.getDeclaration():
            continue

        # code_chunk = ' '.join(code_chunk.getCode().split())
        code_chunk = code_chunk.getCode().split()

        ####
        new_code_chunk = []
        for token in code_chunk:
            if token in keywords:
                continue
            else:
                new_token = ""
                for t in token:
                    if t in punctuation:
                        new_token += ' '
                    else:
                        new_token += t
                if new_token.strip() != "":
                    new_code_chunk.append(' '.join(new_token.split()))
        code_chunk = ' '.join(new_code_chunk)
        ####

        sc_nl_vecs.append( torch.mean ( torch.squeeze ( ebm.embedding ( 'sc_nl', code_chunk ) ), dim=0 ) )
        sc_pl_vecs.append( torch.mean ( torch.squeeze ( ebm.embedding ( 'sc_pl', code_chunk ) ), dim=0 ) )

    sc_nl_vecs = torch.stack(sc_nl_vecs, dim=0) # [#code_chunks, 768] 
    sc_pl_vecs = torch.stack(sc_pl_vecs, dim=0) # [#code_chunks, 768] 

    # print("\nsc_nl_vecs: {}\n".format(sc_nl_vecs.size())) # 그래 코드 #chunks 가 다를 수는 없지 
    # print("sc_pl_vecs: {}\n".format(sc_pl_vecs.size()))   # #tokens in each chunk는 다를 지언정

    # if len(sc_nl_vecs) > 1:
    #     sc_nl_vec = torch.mean(sc_nl_vecs, dim=0) # [768]
    #     sc_pl_vec = torch.mean(sc_pl_vecs, dim=0) # [768]
    # else:
    #     sc_nl_vec = sc_nl_vecs
    #     sc_pl_vec = sc_pl_vecs
    return sc_nl_vecs, sc_pl_vecs

def _sampleFalse(bug_report):
    target_size = len(bug_report.getNewFiles())
    false_candidates = bug_report.false_candidates
    false_files = np.random.choice(false_candidates, target_size, replace=False)
    bug_report.false_code_files = false_files

def setAllCandidates(bug_reports):
    for i in range(len(bug_reports)):
        bug_report = bug_reports[i]
        false_candidates = []
        for j in range(len(bug_reports)):
            if i == j: continue
            false_candidates.extend(bug_reports[j].getNewFiles())
        bug_report.false_candidates = copy.deepcopy(false_candidates)
            
def squeeze(nl, sc_nl, sc_pl):
    if len(nl.shape) == 4:
        nl = torch.squeeze(nl)
    if len(sc_nl.shape) == 4:
        sc_nl = torch.squeeze(sc_nl)
    if len(sc_pl.shape) == 4:
        sc_pl = torch.squeeze(sc_pl)
    return nl.view(-1, 512, 768), sc_nl.view(-1, 512, 768), sc_pl.view(-1, 512, 768)

def makeLabel(labels, bin=False):
    if bin:
        print(labels)
        results = [ torch.LongTensor([label]) for label in labels]
        results = torch.squeeze(torch.stack(results, dim=0)).view(-1)
    else:
        results = torch.FloatTensor([[0, 1]]) if labels == 1 else torch.FloatTensor([[1, 0]])
        # results = [ torch.FloatTensor([0, 1]) if label == 1 else torch.FloatTensor([1, 0]) for label in labels ]
        # results = torch.stack(results, dim=0)
    return results