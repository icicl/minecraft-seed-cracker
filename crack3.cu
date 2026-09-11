#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define LCG_MUL 0x5DEECE66DL
#define LCG_ADD 0xB
#define LCG_MSK 0xFFFFFFFFFFFFL

#define MAX_DISTINCT_ITEMS 16
#define MAX_LOOT_TABLE_SIZE 2048
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
__device__ uint32_t  d_match_count;
__device__ uint32_t  d_correct_count;

uint8_t* h_table;
uint8_t* h_target;
uint8_t  h_distinct_items;
uint8_t  h_popcnt;
uint32_t h_shuffle_order;

__device__ uint8_t can_spawn_desert_pyramid(int64_t seed_with_salt, int32_t mcx, int32_t mcz) {
    int64_t seed = (seed_with_salt ^ LCG_MUL) & LCG_MSK;
    return next_int(seed, 32 - 8) == mcx && next_int(seed, 32 - 8) == mcz;
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

#define FIFODEPTH 1
#define BLOCKSIZE 256

template<uint8_t distinct_items>
__global__ void check_loot_collision(int64_t seed_start, uint64_t num_seeds) {
    __shared__ uint32_t s_table[MAX_LOOT_TABLE_SIZE];
    for (int i=threadIdx.x; i<MAX_LOOT_TABLE_SIZE/4; i+=blockDim.x) s_table[i] = ((uint32_t*)d_table)[i];
    __shared__ uint8_t s_target[distinct_items];
    for (int i=threadIdx.x; i<distinct_items; i+=blockDim.x) s_target[i] = d_target[i];
    __syncthreads();
    int txg = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int8_t loot[distinct_items];

        int32_t x = 7984, z = 10480;
        int spacing = 32, separation = 8, salt = 14357617;
        int32_t cx = x/16, cz = z/16;
        int32_t mcx = cx % spacing, mcz = cz % spacing;
        int32_t rx = cx / spacing, rz = cz / spacing;
        int64_t structure_salt = (rx*341873128712 + rz*132897987541 + salt);

    uint64_t dseed = txg;

    __shared__ int64_t fifo[BLOCKSIZE*FIFODEPTH];
    int64_t* fifo_addr = fifo + tx*FIFODEPTH;
    uint8_t fifo_start = 0, fifo_end = 0, fifo_size = 0;

    while (dseed < num_seeds || __any_sync(0xFFFFFFFF, fifo_size > 0)) {
        for (int i=0; i<4; i++) { // avoid excessive __*sync() calls
            if (dseed < num_seeds && fifo_size < FIFODEPTH) {
                int64_t seed = seed_start + dseed;
                if (can_spawn_desert_pyramid(seed + structure_salt, mcx, mcz)) {
                    fifo_addr[fifo_end % FIFODEPTH] = seed;
                    fifo_end++;
                    fifo_size++;
                }
                dseed += stride;
            }
        }
        if (!__any_sync(0xFFFFFFFF, fifo_size == FIFODEPTH)) continue;
        if (fifo_size == 0) continue;
        int64_t seed = fifo_addr[fifo_start%FIFODEPTH];
        int64_t wseed = seed;
        fifo_start++;
        fifo_size--;
        
        for (int calls=0; calls<4; calls++) { // DES TEMPLE TODO parameterize (use x/z to get call#)
            int64_t feat_seed = get_lcg_feature_seed(seed, x, z, 3, 4, calls) & LCG_MSK;
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
    cudaMemset(&d_match_count, 0, sizeof(uint32_t));
    cudaMemset(&d_correct_count, 0, sizeof(uint32_t));

    printf("Launching kernel...\n");
    const uint64_t NUM_SEEDS_TO_CHECK = 1L << 36;
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
    int popcnt = 0;
    for (int i=0; i<27; i++) popcnt += ((shuffle_order >> i)&1);
    if (distinct_items > MAX_DISTINCT_ITEMS) {
        printf("ERROR - too many items in loot table. Recompile with larger MAX_DISTINCT_ITEMS.\n");
        exit(1);
    }
    // todo: check for loot table too large
    h_table = table;
    h_target = target;
    h_distinct_items = distinct_items;
    h_popcnt = popcnt;
    h_shuffle_order = shuffle_order;

//    print_table(table);
    int seed = 0;
    gpu_init(seed & 0xFFFFFFFF00000000L);

    return 0;
}




