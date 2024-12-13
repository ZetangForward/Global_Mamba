####################################################
########    Configs for MQAR datasets  #############
####################################################
##  train_seed  42 + idx
##  valid_seed  1234 + idx
##  test_seed   5678 + idx
VOCAB_SIZE = 20000



###############################################################
#######################       4k         ######################
###############################################################
train_configs = [
        {"vocab_size": VOCAB_SIZE, "input_seq_len": 2**i, "num_examples": 4000, "num_kv_pairs": 2**x}
        for i in range(6, 13)
        for x in range(i//2-1, i-3)
    ]
valid_configs = [
    {"vocab_size": VOCAB_SIZE, "input_seq_len": 2**i, "num_examples": 100, "num_kv_pairs": 2**x}
    for i in range(8, 14)
    for x in range(i//2-1, i-3)
]
valid_configs.extend([
    {"vocab_size": VOCAB_SIZE, "input_seq_len": (2**i + 2**(i+1))//2, "num_examples": 100, "num_kv_pairs": 2**x}
    for i in range(8, 13)
    for x in range(i//2-1, i-3)
])


###############################################################
#######################  based  v3       ######################
###############################################################
train_based_configs = [    
        {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":100_000, "num_kv_pairs":4},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":20_000, "num_kv_pairs":8},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":20_000, "num_kv_pairs":8},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":20_000, "num_kv_pairs":16},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":20_000, "num_kv_pairs":32},]
    
test_based_configs = [
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":1_000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":1_000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":1_000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":1_000, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":1_000, "num_kv_pairs":64},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":1_000, "num_kv_pairs":128},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":1_000, "num_kv_pairs":256},
]

###############################################################
#######################  based  v2       ######################
###############################################################
train_based_v2_configs = [    
        {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":100_000, "num_kv_pairs":4},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":20_000, "num_kv_pairs":8},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":20_000, "num_kv_pairs":8},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":20_000, "num_kv_pairs":16},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":20_000, "num_kv_pairs":32},]

test_based_v2_configs = [
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":1_000, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":1_000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":1_000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":1_000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":1_000, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":1_000, "num_kv_pairs":64},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":1_000, "num_kv_pairs":128},
]



###############################################################
##################    kvpairs卡128  v3       ##################
##################    fromsk 64的数据2w      ##################
###############################################################
train_kv128_configs_fromsk = [
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":20000, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":20000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":20000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":20000, "num_kv_pairs":16},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":32},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":64},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":64},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":128}, ]


###############################################################
##################    kvpairs卡128  v2       ##################
##################    fromsk 64的数据2w      ##################
###############################################################
train_kv128_v2_configs_fromsk = [
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":20000, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":20000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":20000, "num_kv_pairs":8},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":6},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":8},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":32},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":64},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":10000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":10000, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":10000, "num_kv_pairs":64},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":10000, "num_kv_pairs":128},]

###############################################################
##################    kvpairs卡128  v3       ##################
##################    frompre 64    1w       ##################
###############################################################
train_kv128_configs_frompre = [
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":10000, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":10000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":10000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":10000, "num_kv_pairs":16},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":32},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":64},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":64},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":128}, ]

###############################################################
##################    kvpairs卡128  v3       ##################
##################    frompre 64    1w       ##################
###############################################################
train_kv128_v2_configs_frompre = [
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":10000, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":10000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":10000, "num_kv_pairs":8},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":6},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":10000, "num_kv_pairs":8},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":32},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":64},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":10000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":10000, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":10000, "num_kv_pairs":64},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":10000, "num_kv_pairs":128},]


###############################################################
##################      v2  64k  test        ##################
###############################################################
test_64k_v2_configs = [
    {"vocab_size": VOCAB_SIZE, "input_seq_len": 2**i, "num_examples": 100, "num_kv_pairs": 2**x}
    for i in range(6, 17)
    for x in range(0, i-2)
]

test_64k_v2_configs.extend([
    {"vocab_size": VOCAB_SIZE, "input_seq_len": (2**i+2**(i+1))//2, "num_examples": 100, "num_kv_pairs": 2**x}
    for i in range(6, 16)
    for x in range(0, i-2)
])

###############################################################
##################      v3  64k  test        ##################
###############################################################
test_64k_v3_configs = [
    {"vocab_size": VOCAB_SIZE, "input_seq_len": 2**i, "num_examples": 100, "num_kv_pairs": 2**x}
    for i in range(6, 16)
    for x in range(0, i-1)
]

test_64k_v3_configs.extend([
    {"vocab_size": VOCAB_SIZE, "input_seq_len": (2**i+2**(i+1))//2, "num_examples": 100, "num_kv_pairs": 2**x}
    for i in range(6, 15)
    for x in range(0, i-1)
])

###############################################################
##########  v2  key_len2  value_len4 512     ##################
###############################################################
train_k2v4_512_configs = [
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":40000, "num_kv_pairs":2},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":20000, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":20000, "num_kv_pairs":4},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":8},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":16}, ]

test_k2v4_512_configs = [
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":100, "num_kv_pairs":2},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":100, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":100, "num_kv_pairs":4},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":100, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":100, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":100, "num_kv_pairs":8},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":100, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":100, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":100, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":100, "num_kv_pairs":16}, 
    
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":100, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":100, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":100, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":100, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":100, "num_kv_pairs":32}, ]



###############################################################
###############         v2  context          ##################
###############################################################
train_v2_context_configs = [    
        {"vocab_size":VOCAB_SIZE, "input_seq_len":128,  "context_len":64, "num_examples":10_000, "num_kv_pairs":2},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":128,  "context_len":64, "num_examples":10_000, "num_kv_pairs":4},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":128,  "context_len":64, "num_examples":10_000, "num_kv_pairs":8},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":128,  "context_len":64, "num_examples":10_000, "num_kv_pairs":12},

        {"vocab_size":VOCAB_SIZE, "input_seq_len":256,  "context_len":128, "num_examples":10_000, "num_kv_pairs":4},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":256,  "context_len":128, "num_examples":10_000, "num_kv_pairs":8},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":256,  "context_len":128, "num_examples":10_000, "num_kv_pairs":16},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":256,  "context_len":128, "num_examples":10_000, "num_kv_pairs":24},

        {"vocab_size":VOCAB_SIZE, "input_seq_len":512,  "context_len":256, "num_examples":10_000, "num_kv_pairs":8},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":512,  "context_len":256, "num_examples":10_000, "num_kv_pairs":16},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":512,  "context_len":256, "num_examples":10_000, "num_kv_pairs":32},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":512,  "context_len":256, "num_examples":10_000, "num_kv_pairs":48},
        
        {"vocab_size":VOCAB_SIZE, "input_seq_len":1024,  "context_len":512, "num_examples":10_000, "num_kv_pairs":16},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":1024,  "context_len":512, "num_examples":10_000, "num_kv_pairs":32},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":1024,  "context_len":512, "num_examples":10_000, "num_kv_pairs":64},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":1024,  "context_len":512, "num_examples":10_000, "num_kv_pairs":96},

        {"vocab_size":VOCAB_SIZE, "input_seq_len":2048,  "context_len":1024, "num_examples":10_000, "num_kv_pairs":32},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":2048,  "context_len":1024, "num_examples":10_000, "num_kv_pairs":64},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":2048,  "context_len":1024, "num_examples":10_000, "num_kv_pairs":128},
        {"vocab_size":VOCAB_SIZE, "input_seq_len":2048,  "context_len":1024, "num_examples":10_000, "num_kv_pairs":192},]



###############################################################
###############      v5_context 1024          #################
###############################################################
train_v5_context_configs = [    
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128,  "context_len":64, "num_examples":10_000, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128,  "context_len":64, "num_examples":10_000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128,  "context_len":64, "num_examples":10_000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128,  "context_len":64, "num_examples":10_000, "num_kv_pairs":12},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":256,  "context_len":128, "num_examples":10_000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256,  "context_len":128, "num_examples":10_000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256,  "context_len":128, "num_examples":10_000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256,  "context_len":128, "num_examples":10_000, "num_kv_pairs":24},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":512,  "context_len":256, "num_examples":10_000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512,  "context_len":256, "num_examples":10_000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512,  "context_len":256, "num_examples":10_000, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512,  "context_len":256, "num_examples":10_000, "num_kv_pairs":48},
    
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024,  "context_len":512, "num_examples":10_000, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024,  "context_len":512, "num_examples":10_000, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024,  "context_len":512, "num_examples":10_000, "num_kv_pairs":64},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024,  "context_len":512, "num_examples":10_000, "num_kv_pairs":96},]

test_v5_context_configs = [    
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128,  "context_len":64, "num_examples":100, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128,  "context_len":64, "num_examples":100, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128,  "context_len":64, "num_examples":100, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128,  "context_len":64, "num_examples":100, "num_kv_pairs":12},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":256,  "context_len":128, "num_examples":100, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256,  "context_len":128, "num_examples":100, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256,  "context_len":128, "num_examples":100, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256,  "context_len":128, "num_examples":100, "num_kv_pairs":24},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":512,  "context_len":256, "num_examples":100, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512,  "context_len":256, "num_examples":100, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512,  "context_len":256, "num_examples":100, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512,  "context_len":256, "num_examples":100, "num_kv_pairs":48},
    
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024,  "context_len":512, "num_examples":100, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024,  "context_len":512, "num_examples":100, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024,  "context_len":512, "num_examples":100, "num_kv_pairs":64},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024,  "context_len":512, "num_examples":100, "num_kv_pairs":96},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":2048,  "context_len":1024, "num_examples":100, "num_kv_pairs":32},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":2048,  "context_len":1024, "num_examples":100, "num_kv_pairs":64},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":2048,  "context_len":1024, "num_examples":100, "num_kv_pairs":128},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":2048,  "context_len":1024, "num_examples":100, "num_kv_pairs":196},]




###############################################################
############     mqar-v5_len-k4v8-1024        #################
###############################################################
configs = generate_configs(max_seq_len=1024, num_examples=5000, split='train', dataset_version='v5_len', key_len=4, value_len=8, max_kv_pairs=None, VOCAB_SIZE=20000)
build_dataset(configs=configs, split='train', dataset_version='v5_len', version_name="k4v8", data_name='1024')



###############################################################
############        mqar-v5-k2v4-512          #################
###############################################################
train_k2v4_512_configs = [
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":40000, "num_kv_pairs":2},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":20000, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":20000, "num_kv_pairs":4},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":10000, "num_kv_pairs":8},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":10000, "num_kv_pairs":16}, ]

test_k2v4_512_configs = [
    {"vocab_size":VOCAB_SIZE, "input_seq_len":64, "num_examples":100, "num_kv_pairs":2},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":100, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":128, "num_examples":100, "num_kv_pairs":4},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":100, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":100, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":256, "num_examples":100, "num_kv_pairs":8},

    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":100, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":100, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":100, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":512, "num_examples":100, "num_kv_pairs":16}, 
    
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":100, "num_kv_pairs":2},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":100, "num_kv_pairs":4},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":100, "num_kv_pairs":8},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":100, "num_kv_pairs":16},
    {"vocab_size":VOCAB_SIZE, "input_seq_len":1024, "num_examples":100, "num_kv_pairs":32}, ]