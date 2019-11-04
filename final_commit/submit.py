import time, os
import json
import numpy as np
json_name = "./result.json"
def main():
    result = []
    with open(json_name,'w') as fp:
        json.dump(result, fp, indent = 4, separators=(',', ': '))
        
if __name__ == "__main__":
    main()
