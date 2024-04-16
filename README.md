# URL_RAG

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

A simple project to try RAG, using Langchain

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

Instructions on how to install and set up the project.

It is reccomended to set up a virtual enviroment for the project, and mention versions of libraries that are being used in requirement.txt file.

But I have chosen to be lazy in mentioning the versions.

To run this project you should

* have Ollama installed and running before starting.
* pull the LLM model you want to use in ollama, I have used mistral here, you free to choose.
* In case you also want to use mistral, do  
`ollama pull mistral`, and that should pull the model. Replace mistral with the name of other model if you are using something else.  
Run `pip install -r requirements.txt`.

Starting:
* After you have installed the requirements, If something is missing, Sorry and please add it pip.  
Run `streamlit run app.py`  

## Usage

You can choose How many URLs you want to load by changing the argument of range in line 23 in 'app.py'

Run the app, put in URLs, click process URLs and that shoul start fetching the pages(HTML pages only, for now). Split them, embed and vecotrize and store them.  

Type in question in the text box that appears and click the 'ask' button to ask the question.

That should fetch you the answer.
