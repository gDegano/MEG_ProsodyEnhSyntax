#!/usr/bin/python
"""
Using gpt2 to get word info
@Giulio Degano, Phd
University of Geneva
05/02/2021
"""

import os 
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel 
import spacy
import torch.nn.functional as F
import numpy as np 
import math
import matplotlib.pyplot as plt
#%matplotlib qt

path = 'pathOfTheFolder/CodeStimuliExtraction'


def transform2PoS(IDX,tokenizer):
    optionsPred=[]
    for j in range(IDX.size()[0]):
        predicted_index = IDX[j].item()
        predicted_text = tokenizer.decode([predicted_index])
        x = nlp(predicted_text)
        if len(x)>1:
            optionsPred.append(x[1].pos_)
        else:
            optionsPred.append(x[0].pos_)
    return optionsPred

def gpt2ToSyntax(predictions,univ_PoS):
    #using softmax to get probability values
    PoS_ScoreWord=np.zeros([predictions.size()[1],len(univ_PoS)])
    wordList=[]
    for i in range(predictions.size()[1]):
        if (i % 100)==0:
            print(i)
        softP_step=F.softmax(predictions[0, i, :],dim=0) # from logit 2 prob 
        sortedT, indicesSort = torch.sort(softP_step,descending=True)
        
        cumProb=torch.cumsum(sortedT, dim=0)
        cut90=cumProb[cumProb<=.9]
        cutThresh=40
        if cut90.size()[0]<cutThresh:
            idx2trf=indicesSort[0:cutThresh]
            score2trf=np.array(sortedT[0:cutThresh].tolist())
        else:
            idx2trf=indicesSort[0:cut90.size()[0]] 
            score2trf=np.array(sortedT[0:cut90.size()[0]].tolist())
            
        pos_gpt2=transform2PoS(idx2trf,tokenizer)
        
        interSec=set(pos_gpt2) & set(univ_PoS)
        for val in interSec:
            idxPoSList=[i for i,x in enumerate(pos_gpt2) if x==val]
            idx_UniPoS=univ_PoS.index(val)
            scorePos = np.sum([score2trf[idxPoSList,]])
            PoS_ScoreWord[i,idx_UniPoS]=scorePos
    
        wordList.append(tokenizer.decode([tokens_tensor[:,i].item()]))
    return wordList,PoS_ScoreWord

#def transform2SyntVec():
#    pos_gpt2

univ_PoS=['ADJ','ADP','PUNCT','ADV','AUX','SYM','X','INTJ','CONJ','NOUN','DET','PROPN','NUM','VERB','PART','PRON','SCONJ','SPACE']

# path_2_gs='C:\Program Files\gs\gs9.53.3\bin'

# os.environ['PATH']+=os.pathsep + path_2_gs

#speakerList=['DanielKahneman','JamesCameron','JaneMcGonigal','TomWujec']
speakerList=['DanielKahneman','JamesCameron','JaneMcGonigal','TomWujec']

nlp = spacy.load("en_core_web_sm")

for name in speakerList:

#name=speakerList[0]

    f = open(os.path.join(path,'TedLium',name+'_2010.txt'), "r")
    temp=f.read()
    f.close()

    doc = nlp(temp)
    
    token_texts = []
    flagW=0
    for token in doc:
        if flagW==1:
            token_texts.append(' '+token.text)
        else:
            token_texts.append(token.text)
      
        if token.whitespace_:  # filter out empty strings
            flagW=1
        else:
            flagW=0

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    model.eval()

    maxK=2**10
    aa=tokenizer(token_texts)['input_ids']

    temp=[]
    flagNew=[]
    
    for i in range(len(aa)):
        if len(aa[i])>0:
            for j in range(len(aa[i])):
                temp.append(aa[i][j])
                if j==0:
                    flagNew.append(True)
                else:
                    flagNew.append(False)
        
    tokens_tensorBig = tokenizer.encode(temp, return_tensors="pt")
    windN=math.floor(tokens_tensorBig.size()[1]/maxK)

#%%

    fullData=[]
    overlap=700
    for k in range(int(maxK/2),int(tokens_tensorBig.size()[1]),maxK-overlap):
        print([k-maxK/2,k+maxK/2])
    
        if k+int(maxK/2)>int(tokens_tensorBig.size()[1]):
            topCut=int(tokens_tensorBig.size()[1])
        else:
            topCut=k+int(maxK/2)
    
        tokens_tensor = torch.index_select(tokens_tensorBig, 1, torch.tensor([x for x in range(k-int(maxK/2),topCut)])) 

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        model.to('cuda')
    
        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
        
        wordList,PoS_ScoreWord=gpt2ToSyntax(predictions,univ_PoS)

        fullData.append([wordList,PoS_ScoreWord])
 

       
#y, x = np.mgrid[0:18,0:100]
#plt.pcolor(x, y, np.transpose(PoS_ScoreWord), cmap='gist_heat')
#plt.show()
#transform2SyntVec(pos_gpt2)

#from scipy import signal
#f, Pxx_den =signal.periodogram(PoS_ScoreWord[:,15])
#plt.semilogy(f[1:], Pxx_den[1:])
#plt.show()


    arr = np.array(fullData)
    with open(os.path.join(path,'AlignedTeds',name +'_gpt2.npy'), 'wb') as f:
        np.save(f, arr)
        np.save(f, flagNew)