# -*- coding: utf-8 -*-
def prompt_csv_validation(user_msg):
    sys = ("You are an expert in determining document content. "
    "The user uploaded a CSV file representing gene x cell. "
    "Determine if the first line is the cell names?"
    "The second to last lines represent the numerical values of gene expression levels? "
    "The first column represents the gene names? "
    "The output strict minified JSON only: {\"output\":\"Answer\"}. "
    "The answer must be: 'Yes', 'No', 'Uncertain'.")
    return [{"role": "system", "content": sys},{"role": "user", "content": user_msg}]

def prompt_splitting(user_msg):
    sys = ("You are an expert in data splitting. "
    "The user entered segmentation parameters: validation set. "
    "The output strict minified JSON only: {\"valid_size\":\"Value\". "
    "The answer must be a value between 0-1, or -1 instead input error: 'Ratio between 0-1', '-1'.")
    return [{"role": "system", "content": sys},{"role": "user", "content": user_msg}]

def prompt_train_or_predict(user_msg):
    sys = ("You are a language expert. "
    "Please determine whether the user's next steps are related to training, prediction, or cannot be determined?"
    "The output strict minified JSON only: {\"output\":\"Answer\"}. "
    "The answer must be: 'Train', 'Predict', 'Uncertain'.")
    return [{"role": "system", "content": sys},{"role": "user", "content": user_msg}]

def prompt_train_or_not(user_msg):
    sys = ("You are a language expert. "
    "Please determine whether the statement entered by the user indicates affirmation, negation, or inability to determine. "
    "The output strict minified JSON only: {\"output\":\"Answer\"}. "
    "The answer must be: 'Yes', 'No', 'Uncertain'.")
    return [{"role": "system", "content": sys},{"role": "user", "content": user_msg}]

def prompt_predict_or_not(user_msg):
    sys = ("You are a language expert. "
    "Please determine whether the statement entered by the user indicates affirmation, negation, or inability to determine. "
    "The output strict minified JSON only: {\"output\":\"Answer\"}. "
    "The answer must be: 'Yes', 'No', 'Uncertain'.")
    return [{"role": "system", "content": sys},{"role": "user", "content": user_msg}]

def prompt_statistics(user_msg):
    sys = ("You are a verifier. "
    "Please combine the output results of multiple discriminators and provide the most likely category. "
    "The output strict minified JSON only: {\"output\":\"Category\"}. "
    "The category must be a number.")
    return [{"role": "system", "content": sys},{"role": "user", "content": user_msg}]