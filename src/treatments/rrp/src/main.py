from data import DATA 
from helper import *
from config import *
import random
import time
import json

inputs = ['dtlz2','dtlz3','dtlz4','dtlz5','dtlz6','dtlz7','pom3a','pom3b','pom3c','pom3d','SS-A','SS-B','SS-C','SS-D','SS-E','SS-F','SS-G','SS-H','SS-I','SS-J','SS-K','SS-L','SS-M','SS-N','SS-O']
# inputs = ['dtlz2']
file_output = 'rrp_output.txt'
def rrp(input):
    score = []
    randomSeeds = random.sample(range(15000),20)  
    # randomSeeds=[10000]
    for randomSeed in randomSeeds:
        print('------------------------------------------------------------------------------------------')
        print('SEED: ', randomSeed,"    ",input)
        print('------------------------------------------------------------------------------------------')
        data_new = DATA(the['file'])
        best, _, _ = data_new.branch(randomSeed,input)
        max = 100
        l = []
        for r in best.rows:
            if max > round(r.d2h(input,randomSeed, data_new),3):
                max = round(r.d2h(input,randomSeed,data_new),3)
                l = r.cells
        with open(file_output, 'a') as file:
                file.write('\n-------------------------------- BEST \n' + str(l) + ' \n--------------------------------\n')
        
        score.append(max)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    return score

if __name__ == "__main__":
    
    rrps = {}
    
    start_time_0 = time.time()
    for input in inputs:
        start_time_1 = time.time()
        with open(file_output, 'a') as file:
            file.write('\n_______________________________\nRRP for dataset ' + str(input)+'\n_______________________________\n')


        rrps[input] = {}
        rrps[input]['rrp'] = rrp(input)

        end_time_1 = time.time()
        with open(file_output, 'a') as file:
            file.write('\n*****************\nFINAL\n*****************\n' + str(rrps[input])+'\nTime taken: ' + str(end_time_1 - start_time_1) + ' seconds\n*****************\n\n*****************\n')
            file.write('\n_______________________________\nEND for dataset ' + str(input)+'\n_______________________________\n')

        print(rrps)

    end_time_0 = time.time()
    with open(file_output, 'a') as file:
        file.write('\n*****************\nFINAL RRP OUTPUT ENTIRE DATA LIST\n*****************\n' + str(rrps)+'\nTime taken: ' + str(end_time_0 - start_time_0) + '\n*****************\n\n*****************\n')

    file_path = "rrp_data.json"
    with open(file_path, 'w') as file:
        json.dump(rrps, file, indent=4)
    print(f"Data successfully written to {file_path}")