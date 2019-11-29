from preprocess import PreProcessor

def main():
    pp = PreProcessor()
    s_array, t_array, labels = pp.process_data("../training-PHI-Gold-Set1/")
    pp.create_vocab_dict()
    pp.create_train_set(t_array,labels)

if __name__ == "__main__":
    main()