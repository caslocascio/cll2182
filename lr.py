import pandas as pd 
import numpy as np 
import sys

def cost_function(a, w, h, b_zero, b_one, b_two, n):
    fx = b_zero + (b_one*a) + (b_two*w)
    inner = fx - h
    sqr = inner ** 2
    result = (1/(2*n)) * sqr 

    return result 

def lr(csv, location):

    # reading data [age(years), weight(kg), height(m)]
    df = pd.read_csv(csv, header=None)
    df = df.drop(columns=[2])                          # dropping the height column so its zscore is not calculated

    dfcpy = pd.read_csv(csv, header=None)              # making a df "cpy" so I can later add the height column back in

    df_zscore = (df - df.mean())/df.std()              # calculating the zscore of each column 

    # adding vector column
    s = df_zscore.shape
    intercept = np.ones((s[0],1))
    df_zscore.insert(0, "Intercept", intercept)

    df_zscore.insert(3, "2", dfcpy[2])                 # adding height column back in to df 
    ex_lst = [list(row) for row in df_zscore.values]   # creating a list to iterate over in for loop 
    # [1, age, weight, height]

    # result lst: each line contains α,numiters,bias,bage,bweight
    results = []
    
    # alpha: need to add a tenth learning rate after observation
    alphas = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,0.008]

    n = len(df_zscore)                                 # used in calculations 

    for al in alphas:
        # initialize Betas to 0: bias, b_age, b_weight 
        b_zero = 0
        b_one = 0
        b_two = 0
        for i in range(100):
            for j in range(len(ex_lst)):
                a = ex_lst[j][1]
                w = ex_lst[j][2]
                h = ex_lst[j][3]
                b_zero = b_zero - ((al/n)*(b_zero + (b_one*a) + (b_two*w) - h))
                b_one =  b_one - ((al/n)*(b_zero + (b_one*a) + (b_two*w) - h) * a)
                b_two =  b_two - ((al/n)*(b_zero + (b_one*a) + (b_two*w) - h) * h)

        total_cost = 0
        for k in range(len(ex_lst)):
            a = ex_lst[k][1]
            w = ex_lst[k][2]
            h = ex_lst[k][3]
            c = cost_function(a, w, h, b_zero, b_one, b_two, n)
            total_cost += c

        results.append([al, i+1, b_zero, b_one, b_two])
    
    my_df = pd.DataFrame(results)
    my_df.to_csv(location, header=False, index=False)


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Error: please provide csv filename and output location")

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    lr(input_file, output_file)