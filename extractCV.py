import numpy as np
import cv2
import os
from PIL import Image as imgg
import PIL.Image
from pytesseract import image_to_string
import pytesseract
import collections
from collections import defaultdict
import re
import json
import pandas as pd
import nltk.tokenize as nt
import nltk
import spacy as sp
from pdf2image import convert_from_path
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input director

CV=defaultdict(list) #Dictionary to store the final extracted features
#List to store all the main headines that need to be captured
tokens=list(["experience","skills","technical skills","skill","technical skill","education","work experience","professional experiences","exams and certifications","organisations"
             ,"organizations","awards","projects","project experience","coursework"
            ,"extra curriculars","programming experience","programming languages","employment","other details","other skills","training","co-curriculars","co curricular","courses","leadership experience","leadership experiences","extracurricular","leadership","activity","activities","hobbies","interest","interests","personal information","personal informations","additional"])



def pdf_to_png(filepath):
            pages = convert_from_path(filepath, 500)
            for page in range(len(pages)):
                pages[page].save(filepath.split('/')[-1].split('.')[0]+str(page)+'.jpg', 'JPEG')
            return len(pages),filepath.split('/')[-1].split('.')[0]
    
def image_preprocessing(pages,file_head):
    cv_image=list()
    gray_image=list()
    ostu=list()
    for i in range(pages):
        cv_image.append(cv2.imread(file_head+str(i)+'.jpg'))
        gray_image.append(cv2.cvtColor(cv_image[i], cv2.COLOR_BGR2GRAY))
        ret,th = cv2.threshold(gray_image[i],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        ostu.append((ret,th))
    return cv_image,ostu

def token_identifier(output):
    #print(output)
    output_list=(output.split("\n"))
    #print(output_list)
    output_list1=(output.split(" "))
    if(re.sub(r'[^a-zA-Z.\d\s]', '',output) in tokens):
        #print(output)
        return output
    elif(re.sub(r'[^a-zA-Z.\d\s]', '', output_list[0].lower()) in tokens):
        #print(output_list[0])
        return re.sub(r'[^a-zA-Z.\d\s]', '', output_list[0].lower())
    elif(re.sub(r'[^a-zA-Z.\d\s]', ' ', output_list1[0].lower()) in tokens):
        #print(output_list1[0])
        return re.sub(r'[^a-zA-Z.\d\s]', '', output_list1[0].lower()) 
    elif(len(output_list1)>1 and re.sub(r'[^a-zA-Z.\d\s]', ' ', str(output_list1[0])+" "+str(output_list1[1])).lower() in tokens):
        #print(str(output_list1[0])+str(output_list1[1]))
        return  re.sub(r'[^a-zA-Z.\d\s]', '', str(output_list1[0])+" "+str(output_list1[1])).lower()
    #elif(output_list2[0].lower() in tokens):
    #    return (str(output_list2[0]))
    else:
        return None

def identify_name(output_str):
    #print(output_str)
    output_list=(output_str.split("\n"))
    #print(output_list)
    en = sp.load('en')
    people=str()
    for i in output_list:
        sents = en(i)
        for ee in sents.ents :
            #print(ee,ee.label_)
            if(ee.label_ == 'PERSON' and str(ee).isalnum()==False):
                if(len(str(ee))>1):
                    people=people+' '+str(ee)
    return people

def create_data(head,content):
    CV[head].append(content)

def find_email(string):
    return re.findall('\S+@\S+', string)

def content_extraction(output,boxes,token_positionsY,token_positionsX,last_token):
    kernel=(15,15)
    name=list()
    email=list()
    #print(token_positionsY, token_positionsX)
    for x,y,w,h in boxes:
        cropped_image=output[y:y+h,x:x+w]
        img = imgg.fromarray(cropped_image, 'RGB')
        output_str = pytesseract.image_to_string(img, lang='eng')
        #print(output_str)
        if(output_str!=''):
            ide=token_identifier(output_str)
            #print('ide :',ide)
            if(ide==None):
                #print(output_str)
                dist=100000
                dist1=100000
                xmin=100000
                y_final1=str()
                y_final=str()
                #print("x,y : ",x,y)
                for key1,values1 in token_positionsY.items():
                
                    if(y-values1<=dist and y-values1 >= 0):
                        #print("key : ",key1)
                        if(y-values1<=dist1):
                            dist1=y-values1
                            y_final1=key1
                        #if(x in range(token_positionsX[key1]-kernel[0],token_positionsX[key1]+kernel[1])):
                        if(abs(x-token_positionsX[key1])<=xmin):
                                xmin=abs(x-token_positionsX[key1])
                                #print("key,X,xmin : ",key1,token_positionsX[key1],xmin)
                                dist=y-values1
                                y_final=key1
                             
                #print("Tokens Y_finals",y_final1,y_final)
                if((len(y_final)==0 and y_final1=="other details") or ((y_final)=="other details" and y_final1=="other details")):
                    if(last_token=="other details"):
                        if(len(name)==0):
                            name=identify_name(output_str)
                            create_data('name',name)
                       
                        email=find_email(output_str)
                        if(len(email)!=0):
                            create_data('email',email)
                elif(len(y_final)==0):
                    if(y_final1=="other details"):
                        create_data(last_token,output_str)
    
                    else:
                        create_data(y_final1,output_str)
                       
                else: 
                    create_data(y_final,output_str)
                    
            else:
                output_list=(output_str.split("\n"))
                if(len(output_list)>1 and output_list[0].lower() in tokens):
                        CV[output_list[0].lower()].append(output_str.split('\n', 1)[1])
                        continue
                outputlist1=(output_str.split(" "))
                if(len(outputlist1)>1 and (outputlist1[0].lower() in tokens)):
                        CV[outputlist1[0].lower()].append(output_str.replace(outputlist1[0],''))
                        continue
                elif(len(outputlist1)>1 and (outputlist1[0].lower()+outputlist1[1].lower() in tokens)):
                        CV[outputlist1[0].lower()].append(output_str.replace(outputlist1[0]+' '+outputlist1[1],''))
                        continue
                
                

def token_extraction(output,boxes):
    token_positionsY=dict()
    token_positionsX=dict()
    #pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ymax=0
    last_token=str()
    for x,y,w,h in boxes:
        cropped_image=output[y:y+h,x:x+w]
        img = imgg.fromarray(cropped_image, 'RGB')
        output_str = pytesseract.image_to_string(img, lang='eng')
        if(output_str!=''):
            ide=token_identifier(output_str)
            #print("token",ide)
            if(ide!=None):
                
                token_positionsY[ide]=y
                token_positionsX[ide]=x
    token_positionsY["other details"]=0
    token_positionsX["other details"]=0
    #print(token_positionsY, token_positionsX)
    return token_positionsY, token_positionsX
def last_token(token_positionsY):
    ymax=-100
    last_token=str()
    #print(token_positionsY)
    for key,values in token_positionsY.items():
        if(values>=ymax):
            last_token=token_positionsY[key]
            ymax=values
    return last_token
def process_CV(ostu,cv_image,pages):
    last="other details"
    for i in range(pages):
        output=cv_image[i].copy()
        # assign a rectangle kernel size
        boxes=list() #to store contours
        boundary=list()
        kernel = np.ones((15,15), 'uint8')
        par_img = cv2.dilate(ostu[i][1],kernel,iterations=3)
        (_,contours,_)=(cv2.findContours(par_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE))
        for cnt in contours:
            boxes.append(cnt)
        boxes.reverse()
        for cn in boxes:
            x,y,w,h = cv2.boundingRect(cn)
            boundary.append(list([x,y,w,h]))
            cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),1)
            #cropped_image=output[y:y+h,x:x+w]
            #text_extraction(cropped_image,list([x,y,w,h]))
        cv2.imwrite("output/Bounding_Image.jpg",output)
        token_positionsY, token_positionsX = token_extraction(output,boundary)
        content_extraction(output,boundary,token_positionsY,token_positionsX,last)
        last=last_token(token_positionsY)
    #return output
    
#pages : number of pages in the pdf file
#file_head : the head of the image name which will be followed by the page number.
def CV_extraction(filepath):
    pages,file_head=pdf_to_png(filepath)  
    cv_image,ostu=image_preprocessing(pages,file_head)
    output_CV = process_CV(ostu,cv_image,pages)
    return CV
    



     
     
     
     
     
     