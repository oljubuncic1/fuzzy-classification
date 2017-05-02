from operator import itemgetter
import sys
import ast

curr_k = None
curr_predictions = {}

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    # parse the input we got from mapper.py
    k, v = line.split('\t', 1)

    if k == curr_k:
        # continue
        m = ast.literal_eval(v)
        for label in m:
            if not label in curr_predictions:
                curr_predictions[label] = 0
            
            curr_predictions[label] += m[label]
    else:
        if curr_k is not None:
            # new key arrived, print results
            # for previous
            print(curr_k, "\t", curr_predictions)
            
        curr_k = k
        curr_predictions = m