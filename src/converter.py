import numpy as np
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom

def get_label_positions(labels,idx2tag):
    true_labels = np.where(np.logical_and(labels != 0,labels != 1))
    doc_labels = []
    prev = None
    label_idx = None
    label_tag = None
    for sent,idx in zip(true_labels[0],true_labels[1]):
        curr_idx = labels[sent,idx]
        curr_tag = idx2tag[curr_idx]
        if prev is None: # first iteration
            prev = sent,[idx],curr_tag[2:]
            label_idx = curr_idx
            label_tag = curr_tag
            continue
        if sent == prev[0] and idx-1 == prev[1][-1] and curr_tag[1:] == label_tag[1:] : # consecutive tags
            prev[1].append(idx)
        else:
            doc_labels.append(prev)
            prev = sent,[idx],curr_tag[2:]
            label_idx = curr_idx
            label_tag = curr_tag
    doc_labels.append(prev) # last prev
    return doc_labels # [(row:int,cols:[int],tag:[string])]

def bio_to_i2d2(df,doc_labels,note):
    """
    Params: 
    df: DataFrame with SINGLE document
    labels: Indices of labels: (row,[cols])
    note: Raw note
    """
    # Add tag elements to i2d2_tags list
    i2d2_tags = []
    characters = df["characters"]
    tokens = df["sentence"]
    for i,(row,cols,tag) in enumerate(doc_labels):
        row_chars = eval(characters.iloc[row])
        row_tokens = eval(tokens.iloc[row])
        start = row_chars[cols[0]][0]
        end = row_chars[cols[-1]][1]
        phi_array = row_tokens[cols[0]:cols[-1]+1] 
        phi_text = " ".join(phi_array)
        note_phi = note[start:end]
        if not phi_text == note_phi:
            print("WARNING: mismatch between raw note and database. raw: "+note_phi + " DB: "+phi_text)
        print("Start =",start, "End =",end,"Text =",note_phi, "Type =",tag)
        xml_tag = Element(tag, 
        {
            'id':"P"+str(i),
            "start":str(start),
            "end":str(end),
            "text":phi_text,
            "TYPE":tag,
            "comment":""
        })
        i2d2_tags.append(xml_tag)
    # Create full XML document
    root = Element('deIdi2b2')
    text_child = SubElement(root,'TEXT')
    text_child.text = note
    tags_child = SubElement(root,"TAGS")
    for tag in i2d2_tags:
        SubElement(tags_child,tag.tag,attrib=tag.attrib)
    return root  

def prettify(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")