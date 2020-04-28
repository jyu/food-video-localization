#!/usr/bin/env python
# coding: utf-8

# In[50]:


import json
import random
import os
import spacy
from spacy.util import minibatch, compounding
from pathlib import Path

spacy.require_gpu()
nlp = spacy.load("en_core_web_sm")


# In[2]:


# load video names we stored already
f = open('text_cross_val.txt', 'r')
names_f = list(f.readlines())
names = []
for n in names_f:
    names.append(n.replace("\n", ""))

f.close()
print(names)


# In[3]:


def filter_line(line):
    sounds = [
        '(laughing) ',
        '(speaking in foreign language) ',
        '(in foreign language) ',
        '(upbeat percussion music) ',
        '(fast-paced music) ',
        '(jazzy music) ',
        '(smooth jazz music) ',
        '(string music) ',
        '(classy jazz) ',
        '(funky jazz music) ',
        '(soothing guitar) ',
        '(energetic bass) ',
        '(calming piano) ',
        '(whimsical music) ',
        '(band music) ',
        '(chewing) ',
        '(soothing music) ',
        '(upbeat music) ',
        '(dramatic music) ',
        '(upbeat jazz music) ',
        '(slurping) ',
        '(upbeat electronic music) ',
        '(laughs) ',
        '(intense music) ',
        '- '
    ]
    for s in sounds:
        line = line.replace(s, '')
        
    # Check for (sound) at the beginning
    line_split = line.split(" ")
    first = line_split[0]
    
    # Starts with parenthesis, we want to find end
    if first[0] == "(":
        str_to_remove = [first]
        # Does not end with parenthesis
        if first[-1] != ")":
            for rest in line_split[1:]:
                str_to_remove.append(rest)
                if rest[-1] == ")":
                    break
#         print("'" + ' '.join(str_to_remove) + " ',")
        for rem in str_to_remove:
            line_split.remove(rem)
    
    if len(line_split) == 0:
        return ""
    first = line_split[0]
    
    first = line_split[0]
    # Starts with bracket, we want to find end
    if first[0] == "[":
        str_to_remove = [first]
        # Does not end with bracket
        if first[-1] != "]":
            for rest in line_split[1:]:
                str_to_remove.append(rest)
                if rest[-1] == "]":
                    break
        for rem in str_to_remove:
            line_split.remove(rem)
        
    line = ' '.join(line_split)
    return line


# In[44]:


def tag_line(entities, entity_tags, line):
    orig_line = line
    spacy_entities = []
#     print(line)
    for i in range(len(entities)):
        entity = entities[i]
        tag = entity_tags[i]
        if entity in line:
#             print(entity)
            start = line.find(entity)
            end = start + len(entity)
#             print(start, end)
#             print(line[start], line[end])
            spacy_entities.append((start, end, tag))
            
            # Add filler to not label again
            filler = "-" * len(entity)
            line = line.replace(entity, filler)
#             print(line)
    return ((
        orig_line,
        {"entities": spacy_entities}
    ))
    
tag_line(
    ["Pie 'N Burger", 'fruit pies', 'apple', 'boysenberry', 'pie', 'pies'], 
    ['LOCATION_TAG', 'FOOD_TAG', 'FOOD_TAG', 'FOOD_TAG', 'FOOD_TAG', 'FOOD_TAG'], 
    "My name's Michael Osborn, I'm the owner here at Pie 'N Burger."
)


# In[46]:


def get_data_from_name(name):
    train_data = []

    with open('labels/' + name + '.json') as json_file:
        data = json.load(json_file)
    entities = data['entity_list']
    entity_tags = data['entity_tag_list']

    # print(entities, entity_tags)

    for scene_i in range(len(entities)):
        scene_entities = entities[scene_i]
        scene_tags = entity_tags[scene_i]

        f = open('transcripts/' + name + '/scene_' + str(scene_i) + '.txt')
        lines = list(f.readlines())
        lines = lines[:20]
        for line in lines:
            line = line.replace("\n", "")
            line = filter_line(line)
            train_data.append(tag_line(scene_entities, scene_tags, line))
    return train_data


# In[47]:


video_to_data = {}
i = 0
for name in names:
    i += 1
    data = get_data_from_name(name)
    video_to_data[name] = data
    if i % 5 == 0:
        print(str(i) + '/' + str(len(names)))


# In[105]:


def test_NER(nlp, test_data):
#     print(len(test_data))
    total_loc = 0
    total_food = 0
    tp_loc = 0
    tp_food = 0
    fp_loc = 0
    fp_food = 0
    for example in test_data:
        sent = example[0]
        labels = example[1]['entities']
        food_ents = []
        loc_ents = []
        for l in labels:
            if l[2] == 'FOOD_TAG':
                food_ents.append(sent[l[0]:l[1]])
                total_food += 1
            if l[2] == 'LOCATION_TAG':
                loc_ents.append(sent[l[0]:l[1]]) 
                total_loc += 1
        
#         print(loc_ents)
#         print(food_ents)
        doc = nlp(sent)
        for ent in doc.ents:
#             print(ent.label_, ent.text)
            if ent.label_ == 'FOOD_TAG':
                if ent.text in food_ents:
                    tp_food += 1
                else:
                    fp_food += 1
            if ent.label_ == 'LOCATION_TAG':
                if ent.text in loc_ents:
                    tp_loc += 1
                else:
                    fp_loc += 1
    
    loc_recall = tp_loc / (total_loc)
    loc_precision = tp_loc / (tp_loc + fp_loc)
    food_recall = tp_food / (total_food)
    food_precision = tp_food / (tp_food + fp_food)

    
    print('loc recall', loc_recall)   
    print('loc precision', loc_precision)    

    print('food recall', tp_food / (total_food))   
    print('food precision', tp_food / (tp_food + fp_food))  
    
    if loc_precision + loc_recall == 0:
        loc_f1 = 0
    else:
        loc_f1 = 2 * (loc_precision * loc_recall) / (loc_precision + loc_recall)
        
    if food_precision + food_recall == 0:
        food_f1 = 0
    else:
        food_f1 = 2 * (food_precision * food_recall) / (food_precision + food_recall)
    return (loc_f1 + food_f1) / 2

# nlp = spacy.load('models/ner_0')
# test_NER(nlp, test_data)


# In[103]:


def train_NER(model=None, train_data=[], test_data=[], new_model_name="spacy_ner", output_dir=None, n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    print("Train on ", len(train_data), "Test on ", len(test_data))
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label("LOCATION_TAG")  # add new entity label to entity recognizer
    ner.add_label("FOOD_TAG")  # add new entity label to entity recognizer

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    max_f1 = None

    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            # Train 
            random.shuffle(train_data)
            batches = minibatch(train_data, size=sizes)
            losses = {}
            for batch in batches:
#                 print('batch', batch)
#                 print('unzip', list(zip(*batch)))
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            
            # Test
            print("Itn", itn, "Train loss:", losses['ner'])
            avg_f1 = test_NER(nlp, test_data)
            print("Average test f1:", avg_f1)
#             random.shuffle(test_data)
#             test_losses = {}
#             texts, annotations = zip(*test_data)
#             nlp.update(texts, annotations, sgd=None, losses=test_losses)
            
#             print("Itn", itn, "Train loss:", losses['ner'], "Test loss:", test_losses['ner'])

            if max_f1 == None or avg_f1 > max_f1:
                max_f1 = avg_f1
                if output_dir is not None:
                    output_dir = Path(output_dir)
                    if not output_dir.exists():
                        output_dir.mkdir()
                    nlp.meta["name"] = new_model_name  # rename model
                    nlp.to_disk(output_dir)
                    print("Found best avg_f1", avg_f1, 'saving model to', output_dir)
                
    # test the trained model
#     test_text = "Do you like horses?"
#     doc = nlp(test_text)
#     print("Entities in '%s'" % test_text)
#     for ent in doc.ents:
#         print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
#         output_dir = Path(output_dir)
#         if not output_dir.exists():
#             output_dir.mkdir()
#         nlp.meta["name"] = new_model_name  # rename model
#         nlp.to_disk(output_dir)
#         print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
#         doc2 = nlp2(test_text)
#         for ent in doc2.ents:
#             print(ent.label_, ent.text)


# In[ ]:


# Cross val
k = 5
batches = len(names) // k
val_locs = []
val_foods = []
for i in range(k):
    train_names = []
    test_names = []
    for j in range(i * batches, (i + 1) * batches):
        test_names.append(names[j])
    for name in names:
        if not name in test_names:
            train_names.append(name)

    train_data = []
    for train_name in train_names:
        data = video_to_data[train_name]
        train_data += data
        del data
        
    test_data = []
    for test_name in test_names:
        data = video_to_data[train_name]
        test_data += data
        del data
    train_NER(train_data=train_data,test_data=test_data, output_dir="models/ner" + str(i))

