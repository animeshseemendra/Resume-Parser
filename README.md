# CV- Information Extractor
The following project deals with the difficulty of extraction of useful information from resumes and CV due to different formats of writing them depends on person to person.

## Concept
* The project uses computer vision contour to built a bounding box around the areas where text lies inside a image (converted from pdf file). This ensures easy capture of text from different formats of resumes.
* Since the pdf is converted into an image the project uses teserract to extract text from the contours. 
* Now each bounding box formed would have headline like Experience, Education etc which are called tokens in this project either in the first line of the bounding box or they would have their own seperate box.
* The cordinate (x,y) is used to locate each bounding box text to their correct tokens. Each non tokens will belong to the nearest token present above it and along the x axis distance is calculated to find the nearest token.
* A token named other details is given for the text whose tokens was not found. They are probably the ones which are present above all tokens like names, email etc. 
* The text under other tokens is used extract the name and email id of the person.
* To find the tokens a predefined list is maintained of the probable tokens that are common in most resumes.
* Final output is the Dictionary with keys name as tokens or headings and their values as their text.

## Steps to follow:
* To import the file just use python command import with the name of the file as extractCV
* The file has funtion named CV_extraction which takes CV/resume path as input and outputs a dictionary with headings or tokens as its keys and their belonging text as their values.

## Other Methods inside extractCV
* pdf_to_png(filepath): Takes input path of a pdf file and converts it into an image
* image_preprocessing : Applies ostu threshold in images formed from an image.
* token_identifier : Identifies tokens/headings using mostly regrex.
* identify_name : to identify name
* find_email : to find email
* content_extraction : Assigning text to their proper tokens
* token_extraction : extract tokens and store their cordinates using token_identifier
* last_token : Stores the last token of the page.
* process_CV : Process each image and applies computer vision to find contours and then uses the other methods to perform the whole task.

## Required Libraries
* numpy
* cv2
* os
* PIL 
* pytesseract
* collections
* re
* json
* pandas
* nltk
* spacy
* pdf2image
