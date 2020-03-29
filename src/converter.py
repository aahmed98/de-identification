import numpy as np
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom

# dictionary for tag titles. refer to /docs
hipaa_dic = {
    "PATIENT": "NAME", 
    "DOCTOR": "NAME", 
    "USERNAME": "NAME",
    "HOSPITAL": "LOCATION",  
    "ORGANIZATION": "LOCATION", 
    "STREET": "LOCATION", 
    "CITY": "LOCATION", 
    "STATE": "LOCATION", 
    "COUNTRY": "LOCATION", 
    "ZIP": "LOCATION",
    "LOCATION-OTHER": "LOCATION",
    "PHONE": "CONTACT", 
    "FAX": "CONTACT", 
    "EMAIL": "CONTACT", 
    "URL": "CONTACT", 
    "IPADDRESS": "CONTACT",
    "MEDICALRECORD": "ID", 
    "HEALTHPLAN": "ID",
    "BIOID": "ID",
    "IDNUM": "ID"
}

def get_label_positions(labels,idx2tag):
    """
    Given a list of label indices and an idx2tag dictionary, returns a list of tuples of the form
    (row, cols, tag) for each tag. Cols is a list because tag can span multiple tokens.
    """
    true_labels = np.where(np.logical_and(labels != 0,labels != 1))
    # print(true_labels)
    doc_labels = []
    prev = None
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
    return doc_labels, true_labels # [(row:int,cols:[int],tag:[string])]

def bio_to_i2d2(df,doc_labels,note,true_labels= None):
    """
    Params: 
    df: DataFrame with SINGLE document
    labels: Indices of labels: (row,[cols])
    note: Raw note

    Returns:
    Predicted document in i2b2 format. Used for testing.
    """
    # Add tag elements to i2d2_tags list
    # print(doc_labels)
    # print(df.head(1000))
    i2d2_tags = []
    sentence_groups = df.groupby((['sentence']))
    for i,(row,cols,tag) in enumerate(doc_labels):
        sentence = sentence_groups.get_group(row) # current sentence
        try:
            start = eval(sentence['characters'].iloc[cols[0]]) # (a,b) as string
            end = eval(sentence['characters'].iloc[cols[-1]])
        except IndexError:
            # print("Index Error")
            continue
        start_idx = start[0]
        end_idx = end[1]
        note_phi = note[start_idx:end_idx]
        tag_title = tag
        if tag in hipaa_dic:
            tag_title = hipaa_dic[tag]
        # i2b2 formatting
        xml_tag = Element(tag_title, 
        {
            'id':"P"+str(i),
            "start":str(start_idx),
            "end":str(end_idx),
            "text":note_phi,
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
