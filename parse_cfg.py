def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')     #store the lines in a list
    # print(lines[1])
    lines = [x for x in lines if len(x) > 0] #get read of the empty lines 
    linesx = [x for x in lines if x[0] != '#']
    # print(lines[0][])  
    lines = [x.rstrip().lstrip() for x in linesx]

    # for i,j in zip(lines,linesx):
    #     print("hi")
    #     if i!=j:
    #         print(i)
    
    block = {}
    blocks = []
    
    for line in lines:
        # print(lines[0][1:-1])
        if line[0] == "[":               #This marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks
#    print('\n\n'.join([repr(x) for x in blocks]))
# cfgfile = './yolov3.cfg'
# blocks = parse_cfg(cfgfile)

def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options