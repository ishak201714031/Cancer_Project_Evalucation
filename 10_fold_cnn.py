import json
import os
import re
from keras.callbacks import ModelCheckpoint,EarlyStopping
from lib.data_handler import trim_cases_by_class
from lib.data_handler import balanced_split_list
from lib.data_handler import get_data_token_count
from lib.data_handler import wv_initialize
from lib.data_handler import cnn_tokensToIdx
from lib.data_handler import get_list_unique
from lib.data_handler import balancedCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from lib import basic_cnn
from keras.models import load_model
import numpy as np
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
'''
valid task names:
    gs_behavior_label
    gs_organ_label
    gs_icd_label
    gs_hist_grade_label
    gs_lat_label
'''
# parameters ---------------------------------------------------------------------------
task = 'gs_icd_label'
#test_prop = .1
num_cv = 5
val_prop = .25
preloadedWV=None
min_df = 2  #This means that a term must appear in at least 2 documents to be included in the model's vocabulary. This parameter is useful for removing very rare words or terms that are unlikely to contribute to the model's performance due to their sparse occurrence across the dataset.
pretrained_cnn_name = 'pretrained.h5'
rand_seed = 3545
cnn_seq_len = 1500
# n your parameters, cnn_seq_len is set to 1500, meaning that the CNN is designed to process sequences (e.g., sentences, documents, or any other form of sequential data) of up to 1500 elements (such as words, characters, or time steps) in length.
#For text processing tasks, this could mean that each document or piece of text is either padded or truncated to ensure it has exactly 1500 tokens (words or characters) before being fed into the CNN model. This uniformity in sequence length is important for batch processing in neural networks, allowing the model to efficiently learn from the data.
reverse_seq = True
train_epochs = 50

# Main Function ------------------------------------------------------------

def main(args = None):
    #Initializing a random state with a fixed seed ensures reproducibility. It means that random processes (like data shuffling) are consistent across different runs of the script.
    rand_state = np.random.RandomState(rand_seed)
    #Calls get_task_labels to read data and labels for the specified task and then applies trim_cases_by_class to remove classes with insufficient data.
    #eikhane 951 ta text data and corresponding level ache
    data_label_pairs = get_task_labels(task)
    
    #eikhane shudhu jeishob label er freq shudhu 10 or 10 er upore taderke nise
    data_label_pairs = trim_cases_by_class(data_label_pairs) #eikhane trim kore 942 ta data niche
    #LabelEncoder converts categorical labels into a numerical format required by machine learning models. Setting up cross-validation indices is crucial for evaluating the model's performance across different data subsets.
    label_list = [x[1] for x in data_label_pairs] #eikhane data er shudhu label gulake nitese
    label_encoder = LabelEncoder()
    label_encoder.fit(label_list)
    
    # balancedCV returns a list CVlist where each element is an integer representing the fold number (ranging from 0 to numCV-1) assigned to the corresponding data point in the original dataset. This list facilitates class-balanced, randomly seeded cross-validation by ensuring that each fold has a proportional representation of each class
    #eikhane kon level kon cv te jabe eta ber kore dekhtese
    cv_list = balancedCV(label_list,num_cv,rand_state)
    #Initializes empty lists for storing actual and predicted labels for performance evaluation.
    y_actual,y_pred = [],[]
    
    #Starts a loop over each cross-validation fold
    
    for this_cv in range(num_cv):
        
        #train test split (bujhini)
        #this cv er upore basis kore on index er value gula train and kon gula test e jabe sheta ber kora hoitese
        train_idx = [i for i,cv in enumerate(cv_list) if cv != this_cv]
        test_idx = [i for i,cv in enumerate(cv_list) if cv == this_cv]
        #train idx or test idx er je index gul ase shei index gular data gula amra niboh
        train = [x for i,x in enumerate(data_label_pairs) if i in train_idx]
        test = [x for i,x in enumerate(data_label_pairs) if i in test_idx]
        
        #eikhane train er label gula nitesi
        train_label_list = [x[1] for x in train]
        
        #eikhane amra train set er 25% ke validation set e convert korechi
        train,val = balanced_split_list(train,train_label_list,val_prop)
        
        #eikhane train er prottekta word er document frequency count kora hoche. The intention is to capture the document frequency of each token, not their total occurrence across all documents.
        #eikhane amra prottekta documents mane mane token list niboh. then token list er prottekta token access korboh and then oi token tah shob gula documents e kotobar ase eta ber korboh. 
        
        vocab_counter = get_data_token_count(train)

        #eikahne train er text gula ke word vector e convert kora hoche
        #wvToIdx assigns a unique integer index to every word that has a corresponding word vector. This index is used to locate the word's vector in the word vector matrix (WV_mat). For example, if wvToIdx["apple"] = 3, it means that the vector representing "apple" is stored in the 3rd row of the WV_mat
        wv_mat, wv_to_idx = wv_initialize(preloadedWV,min_df,vocab_counter,rand_state)

        #Tokenizes and converts the train, validation, and test datasets into a format suitable for the CNN. Label encoding is also applied.

        #eikhane amader text ta train tokens er modhe jache and label ta y er modhe jache
        train_tokens,train_y = list(zip(*train))
        
        #cnn_tokensToIdx function is designed to convert a list of tokens (words) into a list of indices based on a mapping provided by wvToIdx, with additional processing like padding, truncating to a maximum length, and optionally reversing the token list
        #eikhane prottekta token ke same length er kortese and jodi na hoy tahole padding kore same kortese
        #wv_to_idx er sathe token er sathe compare kore. jodi token er word ta wv_to_idx e na thake tahole umk name e new ekta index jog kore add kore dey
        #then max lengthe er shoman korar jonno shamne pise 0 add kore padding kore shoman kore dey
        #train_x er modhe amader model e input dewar jinishta ache
        train_x = [cnn_tokensToIdx(x,wv_to_idx,cnn_seq_len,0,reverse_seq) \
                for x in train_tokens]
        # x = the list of tokens it wants to convert
        
        #same
        val_tokens,val_y = list(zip(*val))
        val_x = [cnn_tokensToIdx(x,wv_to_idx,cnn_seq_len,0,reverse_seq) \
                for x in val_tokens]
        #same
        test_tokens,test_y = list(zip(*test))
        test_x = [cnn_tokensToIdx(x,wv_to_idx,cnn_seq_len,0,reverse_seq) \
                for x in test_tokens]
        train_y = label_encoder.transform(train_y)
        test_y = label_encoder.transform(test_y)
        val_y = label_encoder.transform(val_y)
        label_names = get_list_unique(train_y)

        #try to load pretrained model, otherwise re-train
        model_name = '_'.join([task,pretrained_cnn_name])
        
        #Initializes the CNN model, sets up model checkpoints and early stopping, trains the model, and evaluates its performance on the test set using F1 scores.
        #The CNN is initialized and trained on the processed data. Model checkpoints and early stopping are used to save the best-performing model and prevent overfitting. After training, the model is evaluated on the test set, and performance metrics (F1 scores) are calculated. These steps are the core of the machine learning pipeline, where the model learns from the data and its performance is assessed.
        
        cnn=basic_cnn.init_full_network(wv_mat,label_names)
        checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)
        stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
        _ = cnn.fit(x=np.array(train_x),y=np.array(train_y),batch_size=64,epochs=train_epochs,validation_data=tuple((np.array(val_x),np.array(val_y))), callbacks=[checkpointer,stopper])
        top_model = load_model(model_name)
        fold_actual = test_y
        fold_preds_probs = top_model.predict(np.array(test_x))
        fold_preds = [np.argmax(x) for x in fold_preds_probs]
        
        #Final Performance Metrics

        micro_f = f1_score(fold_actual,fold_preds,average = 'micro')
        macro_f = f1_score(fold_actual,fold_preds,average = 'macro')
        print(this_cv,"fold micro-f", micro_f)
        print(this_cv,"fold macro-f", macro_f)
        y_actual.extend(fold_actual)
        y_pred.extend(fold_preds)

    micro_f = f1_score(y_actual,y_pred,average = 'micro')
    macro_f = f1_score(y_actual,y_pred,average = 'macro')
    print("FULL EXPERIMENT micro-f", micro_f)
    print("FULL EXPERIMENT macro-f", macro_f)

def cleanText(text):
    '''
    function to clean text
    '''
    #replace symbols and tokens
    text = re.sub('\n|\r', ' ', text)
    text = re.sub('o clock', 'oclock', text, flags=re.IGNORECASE)
    text = re.sub(r'(p\.?m\.?)','pm', text, flags=re.IGNORECASE)
    text = re.sub(r'(a\.?m\.?)', 'am', text, flags=re.IGNORECASE)
    text = re.sub(r'(dr\.)', 'dr', text, flags=re.IGNORECASE)
    text = re.sub('\*\*NAME.*[^\]]\]', 'nametoken', text)
    text = re.sub('\*\*DATE.*[^\]]\]', 'datetoken', text)
    text = re.sub("\?|'", '', text)
    text = re.sub('[^\w.;:]|_|-', ' ', text)
    text = re.sub('[0-9]+\.[0-9]+','floattoken', text)
    text = re.sub('floattokencm','floattoken cm', text)
    text = re.sub(' [0-9][0-9][0-9]+ ',' largeint ', text)
    text = re.sub('\.', ' . ', text)
    text = re.sub(':', ' : ', text)
    text = re.sub(';', ' ; ', text)

    #lowercase
    text = text.lower()

    #tokenize
    text = text.split()
    return text

def read_json():
    """
    function to read matched_fd.json as list
    """
    with open('adv_500_items.json') as data_file:
        data = json.load(data_file)
    return data

def get_valid_label(task_name,in_data):
    """
    function to get text,labels for valid tasks
    """
    #print(in_data[0])
    valid_entries = [x for x in in_data if x[task_name]['match_status']=="matched"]
    valid_text = [x['doc_raw_text'] for x in valid_entries]
    valid_tokens = [cleanText(x) for x in valid_text]
    valid_labels = [x[task_name]['match_label'] for x in valid_entries]
    return list(zip(valid_tokens,valid_labels)) #it returns 951 valid data

def get_task_labels(in_task):
    read_data = read_json()
    return get_valid_label(in_task,read_data)


if __name__ == "__main__":
    main()
