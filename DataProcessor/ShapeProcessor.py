import numpy as np
import torch
from torch.utils.data import DataLoader
from Models.EmbeddingModel import EmbeddingModel
from DataProcessor.SourceCodeProcessor import readCode
from tqdm import tqdm
import pickle

ebm = EmbeddingModel()

def getDataLoader(bug_reports):
    text_helper_path = './Helper/text_embedding_helper.pkl'
    code_helper_path = './Helper/code_embedding_helper.pkl'
    text_embedding_helper, code_embedding_helper = _loadHelper(text_helper_path, code_helper_path)

    _setAllCandidates(bug_reports)
    for bug_report in tqdm(bug_reports, ncols=70, desc=" Now Embedding: "):
        _sampleFalse(bug_report)
        full_data = []

        if bug_report.getBugId() in text_embedding_helper.keys():
            nl = text_embedding_helper[bug_report.getBugId()]
        else:
            text = bug_report.getSummary() + bug_report.getDescription()
            nl = ebm.embedding('nl', text)
            text_embedding_helper[bug_report.getBugId()] = nl
        
        for file in bug_report.getNewFiles():
            if file in code_embedding_helper.keys():
                sc_nl, sc_pl = code_embedding_helper[file]
            else:
                sc_nl, sc_pl = _codeEmbedding(file)
                if sc_nl is None: continue
                code_embedding_helper[file] = (sc_nl, sc_pl)

            full_data.append( (nl, sc_nl, sc_pl, 1) )

        for file in bug_report.false_code_files:
            if file in code_embedding_helper.keys():
                sc_nl, sc_pl = code_embedding_helper[file]
            else:
                sc_nl, sc_pl = _codeEmbedding(file)
                if sc_nl is None: continue
                code_embedding_helper[file] = (sc_nl, sc_pl)

            full_data.append( (nl, sc_nl, sc_pl, 0) )

    _saveHelper(text_embedding_helper, code_embedding_helper, text_helper_path, code_helper_path)

    dataloader = DataLoader(full_data, batch_size=16, shuffle=True)
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
    if is_error == True: return None, None

    for code_chunk in code_file.getCodeChunks():
        if "Import" in code_chunk.getDeclaration() or "Package" in code_chunk.getDeclaration():
            continue

        code_chunk = ' '.join(code_chunk.getCode().split())

        sc_nl_vecs.append( torch.squeeze ( ebm.embedding ( 'sc_nl', code_chunk ) ) )
        sc_pl_vecs.append( torch.squeeze ( ebm.embedding ( 'sc_pl', code_chunk ) ) )

    sc_nl_vecs = torch.stack(sc_nl_vecs, dim=0) # [#code_chunks, max_len(512), 768]
    sc_pl_vecs = torch.stack(sc_pl_vecs, dim=0) # [#code_chunks, max_len(512), 768]

    if len(sc_nl_vecs) > 1:
        sc_nl_vec = torch.mean(sc_nl_vecs, dim=0)
        sc_pl_vec = torch.mean(sc_pl_vecs, dim=0)
    else:
        sc_nl_vec = sc_nl_vecs
        sc_pl_vec = sc_pl_vecs
    return sc_nl_vec, sc_pl_vec

def _sampleFalse(bug_report):
    target_size = len(bug_report.getNewFiles())
    false_candidates = bug_report.false_condidates
    false_files = np.random.choice(false_candidates, target_size, replace=False)
    bug_report.false_code_files = false_files

def _setAllCandidates(bug_reports):
    for i in range(len(bug_reports)):
        bug_report = bug_reports[i]
        false_candidates = []
        for j in range(len(bug_reports)):
            if i == j: continue
            false_candidates.extend(bug_reports[j].getNewFiles())
        bug_report.false_condidates = false_candidates
            

