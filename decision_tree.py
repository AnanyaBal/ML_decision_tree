import sys
import numpy as np
import csv
from inspection import calc_entropy_majvote_error
# import ipdb

# global nodelist

class Node:
    """
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    """
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None

def read_data(filename):
    all_records=[]
    with open(filename) as file:
        data = csv.reader(file, delimiter="\t")
        for line in data:
            all_records.append(line)

    headings = np.asarray(all_records)[0:1,][0].astype(str)
    all_records = np.asarray(all_records)[1:,].astype(int)
    

    return all_records, headings

def majority_vote(x):
    # ipdb.set_trace()
    try:
        bincount = np.bincount(x)
    except(ValueError):
        ipdb.set_trace()

    if(bincount.shape[0]==1):
        return np.argmax(x)
    else:
        # ipdb.set_trace()
        # print(bincount.shape)
        if(bincount[1]>=bincount[0]):       #case where counts are equal
            return 1
        else:
            return 0


def train(train_data, max_depth, headings, nodelist, ig, split_counter):
    D = np.asarray(train_data)
    root = tree_recurse(D, max_depth, headings, nodelist, ig, split_counter)
    return root


def tree_recurse(data, max_depth, headings, nodelist, ig, split_counter):
    q = Node()
    cols = data.shape[1]
    rows = data.shape[0]
    labels = data[:,cols-1]
    #base condition
    attr = data[:, 0:cols-1]
    labels_=labels.tolist()
    attr_ = attr.tolist()
    # ipdb.set_trace()
    vals_ = set(labels)
    attr_ = set(tuple(i) for i in attr)
    # print(data)
    
    # print('========================')
    if(len(vals_)==0 or len(vals_)==1 or len(attr_)==1 or ig<0 or split_counter>=max_depth or np.where(attr[0,:]>-1)[0].shape[0]==1) :
        # print("enter base")
        q.vote = majority_vote(labels)

    else:
        entropy, error = calc_entropy_majvote_error(data)
        max_ig = -1
        for i in range(attr.shape[1]):

            if attr[0,i] !=-1:
                split_attr = i
                labels = data[:, -1]
                zero_ratio = len(data[np.where(data[:,i]==0)])/len(data[:,i])
                one_ratio = 1-zero_ratio

                # print('zero ratio ', zero_ratio, ' one ratio ', one_ratio)

                label_zero = data[np.where(data[:,i]==0)]
                label_one = data[np.where(data[:,i]==1)]

                if label_zero.shape[0]!=0:
                    bb = entropy - (zero_ratio * calc_entropy_majvote_error(label_zero)[0])
                else:
                    bb = entropy
                
                
                if label_one.shape[0]!=0:
                    cc =bb- (one_ratio * calc_entropy_majvote_error(label_one)[0])
                else:
                    cc = bb
                
                info_gain = cc

                if(info_gain>max_ig):
                    max_ig = info_gain
                    max_ig_idx = i


        nodelist.append(split_attr)

        q.attr = headings[ max_ig_idx ]
        split_right = data[np.where(data[:,max_ig_idx]==0)[0]]
        split_left = data[np.where(data[:,max_ig_idx]==1)[0]]
        import copy
        if(split_left.shape[0]==0):
            split_left = copy.deepcopy(split_right)

        if(split_right.shape[0]==0):
            split_right = copy.deepcopy(split_left)

        split_left[:,max_ig_idx] = -1
        split_right[:,max_ig_idx] = -1
        split_counter+=1    

        q.left = train(split_left, max_depth, headings, nodelist, max_ig, split_counter)
        q.right = train(split_right, max_depth, headings, nodelist, max_ig, split_counter)

    return q

def predict(node, data, headings):
    if node.vote!=None:
        label = node.vote
    else:
        attr = node.attr
        idx = np.where(headings==attr) #.index(attr)
        # ipdb.ipdb()
        if data[attr] == 0:
            next_node = node.right
        elif data[attr] == 1:
            next_node = node.left
        label = predict(next_node, data, headings)
    return label

def calc_err(gt, labels):
    err_counter = 0
    for i in range(gt.shape[1]):
        if(gt[0][i]!=labels[0][i]):
            err_counter+=1
    # ipdb.set_trace()

    return err_counter/(i+1)

def output(input_, output, node):
    input_data,headings = read_data(input_)
    len_attribute = input_data.shape[-1] - 1
    with open(output,'w') as file:
        label = []
        gt = []
        for i in range(input_data.shape[0]):
            example = {}   
            for j in range(len_attribute):
                example[headings[j]] = input_data[i,j]

            # ipdb.set_trace()
            label.append(predict(node, example, headings))
            gt.append(input_data[i,-1])
            file.write(str(label[i]))
            file.write('\n')

    error = calc_err(np.array([gt]),np.array([label]))
    return error


def split(data):
    labels = data[:, -1]
    zero_count = np.where(labels==0)[0].shape[0]    
    one_count = np.where(labels==1)[0].shape[0]   
    return zero_count, one_count


def pprint(node, data, headings, split_counter):
    pipe='|'
    split_counter+=1
    if(node.right):
        attr = np.where(headings==node.attr)[0][0]
        splitd = data[np.where(data[:,attr]==0)[0]]
        # import ipdb
        # ipdb.set_trace()
       
        print(pipe*split_counter, node.attr, '= 0:' ,'['+str(split(splitd)[0]), '0/', split(splitd)[1], '1]')
        pprint(node.right, splitd, headings, split_counter)

    if(node.left):
        attr = np.where(headings==node.attr)[0][0]
        splitd = data[np.where(data[:,attr]==1)[0]]
        print(pipe*split_counter,node.attr, '= 1:' ,'['+str(split(splitd)[0]), '0/', split(splitd)[1], '1]')
        pprint(node.left, splitd, headings, split_counter)
    

def main():
    train_input, test_input, max_depth, train_output, test_output, metrics_out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
    # ipdb.set_trace()
    train_data, headings = read_data(train_input)
    # print(headings)
    test_data, _ = read_data(test_input)
    nodelist=[]
    tree = train(train_data, max_depth, headings, nodelist, ig=1000, split_counter=0)
    # ipdb.set_trace()

    test_data, headings = read_data(test_input)
    error_train = output(train_input,train_output,tree)
    error_test = output(test_input,test_output,tree)

    with open(metrics_out,'w') as file:
        file.write('error(train): {:.6f}'.format(error_train))
        file.write('\n')
        file.write('error(test): {:.6f}'.format(error_test))

    print('['+ str(split(train_data)[0]), '0/', split(train_data)[1], '1]')
    pprint(tree, train_data, headings, split_counter=0)

if __name__ == '__main__':
    main()
