#!/usr/bin/python
"""
Annotating data with spacy
@Giulio Degano, Phd
University of Geneva
05/02/2021
"""

import os 
import spacy
from spacy import displacy
import json
from pathlib import Path
from negspacy.negation import Negex
#import numpy as np

import string


path = 'pathOfTheFolder/CodeStimuliExtraction'

speakerList=['DanielKahneman','JamesCameron','JaneMcGonigal','TomWujec']

nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("negex")

for name in speakerList:

    f = open(os.path.join(path,'TedLium',name+'_2010.txt'), "r")
    temp=f.read()
    f.close()
    
    # Run sapcy and get sentence list
    doc = nlp(temp)
    
    sentence_spans = list(doc.sents)

    # for i in range(0,3):
    #     svg=displacy.render(sentence_spans[i], style="dep", jupyter=False)
    #     file_name=sentence_spans[i].text[0:10] + '.svg'
    #     output_path = Path( "D:/degano/Desktop/WP_Prosody/Code stimuli extraction/images/" + file_name)
    #     output_path.open("w", encoding="utf-8").write(svg)

    #for token in doc[0:2]:
    #    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #            token.shape_, token.is_alpha, token.is_stop)

    dep_info_comp=[]
    CNT=0
    for i in range(len(sentence_spans)):
        
        # classify if declarative or interrogative sentence... using last char "?" to discriminate
        xSent=sentence_spans[i]
        if len(xSent)>40:
            print(name+' '+str(i)+' '+str(len(xSent)))
        
        if xSent[-1].text=='?':
            typeSent=1
            #print(name+' '+str(i))
        else:
            typeSent=0
        if xSent[-1].text=='\n': # if there is an new paragraph skip
            if xSent[len(xSent)-2].text=='?':
                typeSent=1
            else:
                typeSent=0 
                
        
        for k in xSent:
            if not k.text in string.punctuation :
                
                
                if k.dep_ == 'neg':
                    negation_head_tokens = 1
                else:
                    negation_head_tokens = 0
                
                
                if (k.head.i-k.i)<0:
                    flagLeft=1
                elif (k.left_edge.i-k.i)<0:
                    flagLeft=1
                else:
                    flagLeft=0
                    
                # print(k.text)    
                # print((k.head.i-k.i)<0)
                # print((k.left_edge.i-k.i)<0)
                # print(flagLeft)
                
                #print(token.text, token.i - sent.start)
                if xSent[-1].text in string.punctuation:
                    posToEnd=k.i - xSent[-1].i + 1
                else:
                    posToEnd=k.i - xSent[-1].i
                    
                # Get arc lenght... index minus mindex of the head
                arcL=(k.i-k.head.i)    
                # Insert text, PoS tag, NUmber of right dep, number of left dep, arc lenght and type sent
                dep_info_comp.append([k.text,k.pos_,k.n_rights,k.n_lefts,arcL,typeSent,negation_head_tokens,k.dep_,flagLeft,posToEnd,i])
                CNT=CNT+1
    
    x=json.dumps(dep_info_comp, indent=4)

    f2 = open(os.path.join(path,'AlignedTeds',name+'_Spacy2tags_reduce_2010.txt'), "w") 
    f2.write(x) 
    f2.close() 
