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


# In[108]:


print(len(names))


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


# In[167]:


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
    if tp_loc + fp_loc == 0:
        loc_precision = 0
    else:
        loc_precision = tp_loc / (tp_loc + fp_loc)
        
    food_recall = tp_food / (total_food)
    if tp_food + fp_food == 0:
        food_precision = 0
    else:
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
    return loc_recall, food_f1

# nlp = spacy.load('models/ner_0')
# test_NER(nlp, test_data)


# In[173]:


def train_NER(model=None, train_data=[], test_data=[], new_model_name="spacy_ner", output_dir=None, n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    print("Train on ", len(train_data), "Test on ", len(test_data))
    
    loc_output_dir = output_dir+"_loc"
    food_output_dir = output_dir+"_food"
 
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
    
    max_loc = None
    max_food = None

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
            loc, food = test_NER(nlp, test_data)
            print("Average test val loc recall:", loc, "val food f1", food)
            
            if max_loc == None or loc > max_loc:
                max_loc = loc
                if output_dir is not None:
                    output_dir = Path(loc_output_dir)
                    if not output_dir.exists():
                        output_dir.mkdir()
                    nlp.meta["name"] = new_model_name  # rename model
                    nlp.to_disk(output_dir)
                    print("Found best val loc recall", loc, 'saving model to', output_dir)
                    
            if max_food == None or food > max_food:
                max_food = food
                if output_dir is not None:
                    output_dir = Path(food_output_dir)
                    if not output_dir.exists():
                        output_dir.mkdir()
                    nlp.meta["name"] = new_model_name  # rename model
                    nlp.to_disk(output_dir)
                    print("Found best val food f1", food, 'saving model to', output_dir) 
                    
    # save model to output directory
    if output_dir is not None:

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names


# In[174]:


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
        data = video_to_data[test_name]
        test_data += data
        del data
    train_NER(train_data=train_data,test_data=test_data, output_dir="models/ner" + str(i))


# In[156]:


def eval_NER(nlp, test_data):
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
    
    return tp_loc, fp_loc, total_loc, tp_food, fp_food, total_food

# nlp = spacy.load('models/ner' + str(0))
# for name in ['1_Sushi_Vs_133_Sushi_•_Japan', '5_Pie_Vs_250_Pie', '350_Soup_Vs_29_Soup_•_Taiwan', '1_Bagel_vs_1000_Bagel', '8_Toast_Vs_20_Toast', '9_Fish_Vs_140_Fish', '1_Cookie_Vs_90_Cookie', '10_Noodles_Vs_94_Noodles', '11_Salad_Vs_95_Salad']:
#     data = video_to_data[name]
#     print(eval_NER(nlp, data))

# test_data = []


# In[175]:


# Evaluate results
k = 5
batches = len(names) // k
all_tp_loc, all_fp_loc, all_total_loc, all_tp_food, all_fp_food, all_total_food = 0,0,0,0,0,0
for i in range(k):
    train_names = []
    test_names = []
    for j in range(i * batches, (i + 1) * batches):
        test_names.append(names[j])
    for name in names:
        if not name in test_names:
            train_names.append(name)
            
    test_data = []
    for test_name in test_names:
        data = video_to_data[test_name]
        test_data += data
        del data
    print(test_names)
    print(len(test_data))
    food_nlp = spacy.load('models/ner' + str(i) + '_food')
    loc_nlp = spacy.load('models/ner' + str(i) + '_loc')

    tp_loc, fp_loc, total_loc, _, _, _ = eval_NER(loc_nlp, test_data)
    _, _, _, tp_food, fp_food, total_food = eval_NER(food_nlp, test_data)

    print(tp_loc, fp_loc, total_loc, tp_food, fp_food, total_food)
    all_tp_loc += tp_loc
    all_fp_loc += fp_loc
    all_total_loc += total_loc
    all_tp_food += tp_food
    all_fp_food += tp_food
    all_total_food += total_food
    
print("Loc recall:", all_tp_loc / (all_total_loc))
print("Loc precision:", all_tp_loc / (all_tp_loc + all_fp_loc))
print("Food recall:", all_tp_food / (all_total_food))
print("Food precision:", all_tp_food / (all_tp_food + all_fp_food))


# In[186]:


def save_NER(food_nlp, loc_nlp, name):
    with open('labels/' + name + '.json') as json_file:
        label_data = json.load(json_file)
    res = {}
    for scene_i in range(3):
        data = []

        f = open('transcripts/' + name + '/scene_' + str(scene_i) + '.txt')
        lines = list(f.readlines())
        lines = lines[:20]
        for line in lines:
            line = line.replace("\n", "")
            line = filter_line(line)
            data.append(line)
    
        locations = []
        foods = []
        for line in data:
            doc = food_nlp(line)
            for ent in doc.ents:
                if ent.label_ == 'FOOD_TAG':
                    foods.append(ent.text)
            
            doc = loc_nlp(line)
            for ent in doc.ents:
                if ent.label_ == 'LOCATION_TAG':
                    locations.append(ent.text)
        print(name, scene_i)
#         print('Foods:', foods)
        print('Locations:', locations)
        res[scene_i] = {
            'foods': foods,
            'locations': locations
        }
#         print("Entity list:", label_data['entity_list'][scene_i])
#         print("Entity tag list:", label_data['entity_tag_list'][scene_i])
    with open('cnn_text_preds/' + name + '.json', 'w') as outfile:
        json.dump(res, outfile)


# In[187]:


# Save results from NER
k = 5
batches = len(names) // k
for i in range(k):
    test_names = []
    for j in range(i * batches, (i + 1) * batches):
        test_names.append(names[j])
    food_nlp = spacy.load('models/ner' + str(i) + '_food')
    loc_nlp = spacy.load('models/ner' + str(i) + '_loc')

    for test_name in test_names:
        save_NER(food_nlp, loc_nlp, test_name)

