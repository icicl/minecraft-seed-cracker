#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

enum structures {
    DESERT_TEMPLE,
    BURIED_TREASURE
};

#define LCG_MUL 0x5DEECE66DL
#define LCG_ADD 0xB
#define LCG_MSK 0xFFFFFFFFFFFFL

#define MAX_DISTINCT_ITEMS 16
#define MAX_LOOT_TABLE_SIZE 2048
#define MAX_SPAWN_CHECKS 8
#define ENCH_PACKED 0b11001111100111101111111101110111111L // which enchantments need a second call

__device__ int64_t next(int64_t& seed, int bits) {
    seed = (seed * LCG_MUL + LCG_ADD) & LCG_MSK;
    return seed >> (48 - bits);
}

__device__ int32_t next_int(int64_t& seed, int32_t bound) {
    if ((bound & -bound) == bound) {
        return (bound * next(seed, 31)) >> 31; // pow of 2
    }
    int32_t bits;
    int32_t val;
    do {
        bits = next(seed, 31);
        val = bits % bound;
    } while (bits-val+(bound-1)<0);
    return val;
}

__device__ int32_t next_int_util(int64_t& seed, int32_t min, int32_t max) {
    return min >= max ? min : min + next_int(seed, max-min+1);
}

__device__ int64_t next_long(int64_t& seed) {
    return ((int64_t)next(seed, 32) << 32) + (int64_t)(int32_t)next(seed, 32);
}

__constant__ uint8_t d_table[MAX_LOOT_TABLE_SIZE]; // Adjust size to match your maximum table byte-size
__constant__ uint8_t d_target[MAX_DISTINCT_ITEMS];
__constant__ uint8_t d_distinct_items;
__constant__ uint8_t d_popcnt;
__constant__ uint32_t d_shuffle_order;
__constant__ int32_t d_feature_seed_info[4];
__device__ uint32_t  d_match_count;
__device__ uint32_t  d_correct_count;
__constant__ int64_t  d_spawn_checks[MAX_SPAWN_CHECKS*4];
__constant__ uint8_t d_num_spawn_checks;

uint8_t* h_table;
uint8_t* h_target;
uint8_t  h_distinct_items;
uint8_t  h_popcnt;
uint32_t h_shuffle_order;
int32_t* h_feature_seed_info;
int64_t* h_spawn_checks;
uint8_t h_num_spawn_checks;

__device__ uint8_t can_spawn_desert_pyramid(int64_t seed_with_salt, int32_t mcx, int32_t mcz) {
//    printf("%ld %d %d\n", seed_with_salt, mcx, mcz);
    int64_t seed = (seed_with_salt ^ LCG_MUL) & LCG_MSK;
    return next_int(seed, 32 - 8) == mcx && next_int(seed, 32 - 8) == mcz;
}

__device__ uint8_t can_spawn_buried_treasure(int64_t seed_with_salt) {
    int64_t seed = (seed_with_salt ^ LCG_MUL) & LCG_MSK;
    return next(seed, 24) < 167773; // 0.01*(2^24) = 167772.2
}

__device__ uint8_t spawn_checks(int64_t wseed, int64_t* s_spawn_checks) {
    for (int i=0; i<4*d_num_spawn_checks; i+=4) {
        uint64_t  packed_info = s_spawn_checks[i+3];
        uint8_t structure_type = packed_info >> (48 + 2*6), mx = (packed_info >> (48 + 6)) & (0b111111), mz = (packed_info >> 48) & 0b111111;
//        if (wseed != 777) return 0;// printf("%ld %ld %ld %ld\n", d_spawn_checks[i+0], d_spawn_checks[i+1], d_spawn_checks[i+2], d_spawn_checks[i+3]);
        switch (structure_type) {
            case DESERT_TEMPLE:
//            if (wseed == 777) printf("%ld %ld %ld %ld\n", d_spawn_checks[i+0], d_spawn_checks[i+1], d_spawn_checks[i+2], d_spawn_checks[i+3]);
                if (!can_spawn_desert_pyramid(wseed + packed_info, mx, mz)) return 0;
                break;
            case BURIED_TREASURE:
                if (!can_spawn_buried_treasure(wseed + packed_info)) return 0;
                break;
        }
    }
    return 1;
}


__device__ int64_t get_lcg_feature_seed(int64_t world_seed, int32_t x, int32_t z, int index, int step, int calls) {
    x = (x / 16) * 16;
    z = (z / 16) * 16;
    int64_t seed = (world_seed ^ LCG_MUL) & LCG_MSK;

    int64_t a = next_long(seed) | 1, b = next_long(seed) | 1;
    int64_t dec_seed = (x*a + z*b ^ world_seed) & LCG_MSK;
    int64_t feature_seed = (dec_seed + index + 10000*step) & LCG_MSK;
    feature_seed ^= LCG_MUL;
    for (int _=0; _<calls; _++) next_long(feature_seed);
    return next_long(feature_seed); 
}

#define STACKDEPTH 8
#define BLOCKSIZE 256

template<uint8_t distinct_items>
__global__ void check_loot_collision(int64_t seed_start, uint64_t num_seeds) {
    __shared__ uint32_t s_table[MAX_LOOT_TABLE_SIZE];
    for (int i=threadIdx.x; i<MAX_LOOT_TABLE_SIZE/4; i+=blockDim.x) s_table[i] = ((uint32_t*)d_table)[i];
    __shared__ uint8_t s_target[distinct_items];
    for (int i=threadIdx.x; i<distinct_items; i+=blockDim.x) s_target[i] = d_target[i];
    __shared__ int64_t s_spawn_checks[4*MAX_SPAWN_CHECKS];
    for (int i=threadIdx.x; i<4*MAX_SPAWN_CHECKS; i+=blockDim.x) s_spawn_checks[i] = d_spawn_checks[i];
    __syncthreads();
    int txg = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int8_t loot[distinct_items];

    const int32_t x = d_feature_seed_info[0], z = d_feature_seed_info[1], index = d_feature_seed_info[2], step = d_feature_seed_info[3];


    uint64_t dseed = txg;

    __shared__ int64_t stack[BLOCKSIZE*STACKDEPTH];
//    int64_t* fifo_addr = fifo + tx*FIFODEPTH;
    uint8_t stack_ptr = 0;//, fifo_end = 0, fifo_size = 0;

    
    int64_t seed, wseed;
    while (dseed < num_seeds || stack_ptr > 0) {
        for (int i=0; i<576 && dseed < num_seeds; i++) {
            wseed = seed = seed_start + dseed;
            dseed += stride;
            if (spawn_checks(seed, s_spawn_checks)) {
                stack[STACKDEPTH*tx + stack_ptr] = wseed;
                stack_ptr++;
                if (stack_ptr == STACKDEPTH) break;
            }
        }
        if (stack_ptr == 0) continue;
        stack_ptr--;
        
        for (int calls=0; calls<4; calls++) { // DES TEMPLE TODO parameterize (use x/z to get call#)
            wseed = seed = stack[STACKDEPTH*tx + stack_ptr];
            int64_t feat_seed = get_lcg_feature_seed(seed, x, z, index, step, calls) & LCG_MSK;
            seed = feat_seed ^ LCG_MUL;

            for (int i=1;i<distinct_items; i++) loot[i] = s_target[i];
            loot[0] = INT8_MAX; // don't want to break if 'empty' (id=0) rolled
            uint32_t* table_idx = s_table; // aligned to 4-bytes:    [num_pools | pad | pad | pad] { [rmin | rmax | totweight(2-byte) ] { [item_id | qmin | qmax | ench] } }
            int num_pools = table_idx[0] & 0xFF;
            table_idx += 1;
            int match = 1;
            for (int pool_idx=0; (pool_idx<num_pools)&&match; pool_idx++) {
                uint32_t pool_stats = table_idx[0];
                int rmin = pool_stats & 0xFF, rmax = (pool_stats >> 8) & 0xFF;
                int tot_weight = pool_stats >> 16;
                table_idx += 1;
                int num_rolls = next_int_util(seed, rmin, rmax);
                for (int roll_num=0; roll_num<num_rolls; roll_num++) {
                    int ent_idx = next_int_util(seed, 1, tot_weight)-1;
                    uint32_t entry_stats = table_idx[ent_idx];
                    int qmin = (entry_stats >> 8) & 0xFF, qmax = (entry_stats >> 16) & 0xFF, ench = (entry_stats >> 24);
                    if (ench) {
                        int ench_idx = next_int(seed, 37);
                        if ((ENCH_PACKED >> ench_idx) & 1) next_int(seed, 1); // ignore result (already have enough info w/o it)
                    }
                    int qty = next_int_util(seed, qmin, qmax);
                    if ((loot[entry_stats & 0xFF] -= qty) < 0) {
                        match = 0;
                        break;
                    };
                }
                table_idx += tot_weight;
            }
            for (int i=1; (i<distinct_items)&&match; i++) { // start at 1 (ignore empty)
                if (loot[i] != 0) match = 0;
            }

            if (match) {
                atomicAdd(&d_match_count, 1);
                uint8_t ind[27];
                for (int i=0; i<27; i++) ind[i] = i;
                for (int i=27; i>1; i--) {
                    int j = next_int(seed, i);
                    uint8_t tmp = ind[j];
                    ind[j] = ind[i-1];
                    ind[i-1] = tmp;
                }
                int8_t correct = 1;
                for (int i=27-d_popcnt; i<27; i++) {
                    if (((d_shuffle_order >> ind[i]) & 1) == 0) correct = 0;
                }
                if (correct) {
                    printf("%ld\n", wseed);
                    atomicAdd(&d_correct_count, 1);
                }
            }
        }
    }
}

void print_table(uint8_t* table) { //TODO - fix for new structure
    int idx=1;
    for (int pool_idx=0; pool_idx<table[0]; pool_idx++) {
        printf("[%d, %d]\n", table[idx+0], table[idx+1]);
        idx += 2;
        int num_ent = table[idx++];
        for (int ent_idx=0; ent_idx<num_ent; ent_idx++) {
            printf("    Item #%d (w=%d) - [%d, %d] ench=%d\n", table[idx+0], table[idx+1], table[idx+2], table[idx+3], table[idx+4]);
            idx += 5;
        }
    }
    printf("\n");
}

void gpu_init(int64_t seed) {
    printf("Copying data to GPU...\n");
    cudaMemcpyToSymbol(d_table, h_table, MAX_LOOT_TABLE_SIZE*sizeof(uint8_t));
    cudaMemcpyToSymbol(d_target, h_target, MAX_DISTINCT_ITEMS*sizeof(uint8_t));
    cudaMemcpyToSymbol(d_distinct_items, &h_distinct_items, sizeof(uint8_t));
    cudaMemcpyToSymbol(d_popcnt, &h_popcnt, sizeof(uint8_t));
    cudaMemcpyToSymbol(d_shuffle_order, &h_shuffle_order, sizeof(uint32_t));
    cudaMemcpyToSymbol(d_num_spawn_checks, &h_num_spawn_checks, sizeof(uint8_t));
    cudaMemcpyToSymbol(d_spawn_checks, h_spawn_checks, 4*sizeof(int64_t)*MAX_SPAWN_CHECKS);
    cudaMemcpyToSymbol(d_feature_seed_info, h_feature_seed_info, 4*sizeof(int32_t));
    cudaMemset(&d_match_count, 0, sizeof(uint32_t));
    cudaMemset(&d_correct_count, 0, sizeof(uint32_t));

    printf("Launching kernel...\n");
    const uint64_t NUM_SEEDS_TO_CHECK = 1L << 38;
    #define CASE(n) case n: check_loot_collision<n><<<1024, BLOCKSIZE>>>(seed, NUM_SEEDS_TO_CHECK); break;
    switch (h_distinct_items) {
        CASE(2);
        CASE(3);
        CASE(4);
        CASE(5);
        CASE(6);
        CASE(7);
        CASE(8);
        CASE(9);
        CASE(10);
        CASE(11);
        CASE(12);
        CASE(13);
        CASE(14);// TODO - parameterize discretely
        CASE(15);
        CASE(16);
        default: printf("ERROR - input has more distinct items than this binary supports. Please recompile.\n"); exit(1);
    }

    cudaDeviceSynchronize();

    uint32_t h_match_count;
    uint32_t h_correct_count;
    cudaMemcpyFromSymbol(&h_match_count, d_match_count, sizeof(uint32_t));
    cudaMemcpyFromSymbol(&h_correct_count, d_correct_count, sizeof(uint32_t));
    printf("Checked %lu seeds.\nFound %u matches using items.\nFiltered to %u matches using the shuffle of empty slots.\n", NUM_SEEDS_TO_CHECK, h_match_count, h_correct_count);
}

int main() {
uint32_t shuffle_order = 0b011100111011101100001001101;
uint8_t table[1140] = {2, 0, 0, 0, 2, 4, 232, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 2, 2, 7, 0, 2, 2, 7, 0, 2, 2, 7, 0, 2, 2, 7, 0, 2, 2, 7, 0, 2, 2, 7, 0, 2, 2, 7, 0, 2, 2, 7, 0, 2, 2, 7, 0, 2, 2, 7, 0, 2, 2, 7, 0, 2, 2, 7, 0, 2, 2, 7, 0, 2, 2, 7, 0, 2, 2, 7, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 1, 4, 6, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 3, 1, 3, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 4, 3, 7, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 4, 4, 50, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 4, 1, 8, 0, 4, 1, 8, 0, 4, 1, 8, 0, 4, 1, 8, 0, 4, 1, 8, 0, 4, 1, 8, 0, 4, 1, 8, 0, 4, 1, 8, 0, 4, 1, 8, 0, 4, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 7, 1, 8, 0, 7, 1, 8, 0, 7, 1, 8, 0, 7, 1, 8, 0, 7, 1, 8, 0, 7, 1, 8, 0, 7, 1, 8, 0, 7, 1, 8, 0, 7, 1, 8, 0, 7, 1, 8, 0};
uint8_t target[8] = {0, 0, 4, 3, 5, 1, 1, 6};
int distinct_items = 8;
int64_t spawns_checks[12] = {7353, 10953, BURIED_TREASURE, 0, 7984, 10480, DESERT_TEMPLE, 0, 7450, 10968, DESERT_TEMPLE};
int num_spawn_checks = 3;
int32_t feature_seed_info[4] = {7984, 10480, 3, 4};
    int popcnt = 0;
    for (int i=0; i<27; i++) popcnt += ((shuffle_order >> i)&1);
    if (distinct_items > MAX_DISTINCT_ITEMS) {
        printf("ERROR - too many items in loot table. Recompile with larger MAX_DISTINCT_ITEMS.\n");
        exit(1);
    }
    for (int i=0; i<4*num_spawn_checks; i+=4) {
        int32_t x = spawns_checks[i+0], z = spawns_checks[i+1];
        uint64_t structure_type = spawns_checks[i+2];
        int32_t cx = x/16, cz = z/16;
        if (structure_type == DESERT_TEMPLE) {
            int spacing = 32, separation = 8, salt = 14357617;
            uint64_t mcx = cx % spacing, mcz = cz % spacing;
            int32_t rx = cx / spacing, rz = cz / spacing;
            int64_t structure_salt = (rx*341873128712 + rz*132897987541 + salt);
            spawns_checks[i+0] = mcx;
            spawns_checks[i+1] = mcz;
            spawns_checks[i+3] = (structure_salt & LCG_MSK) | (mcz << 48) | (mcx << (48 + 6)) | (structure_type << (48 + 2*6)); // packed info. LCG seed only needs 48-bits
        } else if (structure_type == BURIED_TREASURE) {
            int salt = 10387320;
            int64_t structure_salt = (cx*341873128712 + cz*132897987541 + salt);
            spawns_checks[i+3] = (structure_salt & LCG_MSK) | (structure_type << (48 + 2*6));
        } else {
            printf("WARNING - unknowns structure type for spawn checking given at x=%d, z=%d.\n", x, z);
        }
    }
    // todo: check for loot table too large
    h_table = table;
    h_target = target;
    h_distinct_items = distinct_items;
    h_popcnt = popcnt;
    h_shuffle_order = shuffle_order;
    h_num_spawn_checks = num_spawn_checks;
    h_spawn_checks = spawns_checks;
    h_feature_seed_info = feature_seed_info;

//    printf("%ld %ld %ld %ld\n", h_spawn_checks[0], h_spawn_checks[1], h_spawn_checks[2], h_spawn_checks[3]);

//    print_table(table);
    int seed = 0;
    gpu_init(seed & 0xFFFFFFFF00000000L);

    return 0;
}




