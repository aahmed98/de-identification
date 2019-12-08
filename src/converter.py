import numpy as np

def get_label_positions(df,docid,labels):
    true_labels = np.where(np.logical_and(labels != 0,labels != 1))
    doc_labels = []
    prev = None
    for sent,idx in zip(true_labels[0],true_labels[1]):
        if prev is None: # first iteration
            prev = sent,[idx]
            continue
        if sent == prev[0] and idx-1 == prev[1][-1]: # consecutive
            prev[1].append(idx)
        else:
            doc_labels.append(prev)
            prev = sent,[idx]
    doc_labels.append(prev) # last prev
    # doc_labels = []
    # running = False
    # current_label = []
    # for i in range(len(labels)): #sentences
    #     sentence_labels = []
    #     for j in range(len(labels[i])): # labels
    #         label = labels[i][j]
    #         if label not in {0,1}: # not PAD or O
    #             running = True
    #             current_label.append(j)
    #         else:
    #             if running:
    #                 sentence_labels.append(current_label)
    #                 current_label = []
    #                 running = False
    #     if running: # reset if last label is not in {0,1}
    #         sentence_labels.append(current_label)
    #         current_label = []
    #         running = False
    #     doc_labels.append(sentence_labels)
    # print(len(doc_labels))
    return doc_labels

def bio_to_i2d2(df,doc_labels,note):
    """
    Params: 
    df: DataFrame with SINGLE document
    labels: Indices of labels: (row,[cols])
    note: Raw note
    """
    i2d2_tags = []
    characters = df["characters"]
    tokens = df["sentence"]
    for row,cols in doc_labels:
        row_chars = eval(characters[row])

        start = row_chars[cols[0]][0]
        end = row_chars[cols[-1]][1]
        print(start)
        print(end)
        text = note[start:end]
        print(text)





