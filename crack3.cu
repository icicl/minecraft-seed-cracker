#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define LCG_MUL 0x5DEECE66DL
#define LCG_ADD 0xB
#define LCG_MSK 0xFFFFFFFFFFFFL

__device__ int64_t next(int64_t* seed, int bits) {
    *seed = (*seed * LCG_MUL + LCG_ADD) & LCG_MSK;
    return (uint64_t)*seed >> (48 - 31);
}

__device__ int32_t next_int(int64_t* seed, int32_t bound) {
    if ((bound & -bound) == bound) {
        return (bound * next(seed, 31)) >> 31; // pow of 2
    }
    int64_t bits;
    int32_t val;
    do {
        bits = next(seed, 31);
        val = bits % bound;
    } while (bits-val+(bound-1)<0);
    return val;
}

__device__ int32_t next_int_util(int64_t* seed, int32_t min, int32_t max) {
    return min >= max ? min : min + next_int(seed, max-min+1);
}

uint8_t* h_table;
uint8_t* h_target;
uint8_t  h_distinct_items;

#define MAX_DISTINCT_ITEMS 64
#define MAX_LOOT_TABLE_SIZE 256

__constant__ uint8_t d_table[MAX_LOOT_TABLE_SIZE]; // Adjust size to match your maximum table byte-size
__constant__ uint8_t d_target[MAX_DISTINCT_ITEMS];
__constant__ uint8_t d_distinct_items;

__global__ void check_loot_collision(int64_t seed_start, uint64_t num_seeds) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint64_t dseed=tx; dseed<num_seeds; dseed+=stride) {
        int64_t seed = seed_start+dseed;
        int loot[256] = {0}; // todo - size smartly. needs only distinct_items length
        uint8_t* table_idx = d_table;
        int num_pools = table_idx[0];
        table_idx += 1;
        for (int pool_idx=0; pool_idx<num_pools; pool_idx++) {
            int rmin = table_idx[0], rmax = table_idx[1];
            int num_ent = table_idx[2];
            table_idx += 3;
            int tot_weight = 0;
            for (int ent_idx=0; ent_idx<num_ent; ent_idx++) tot_weight += table_idx[5*ent_idx + 1];
            int num_rolls = next_int_util(&seed, rmin, rmax);
            for (int roll_num=0; roll_num<num_rolls; roll_num++) {
                int cur_weight = 0, ent_idx=-1;
                int targ_weight = next_int_util(&seed, 1, tot_weight);
                do {cur_weight += table_idx[5*(++ent_idx) + 1];} while (cur_weight < targ_weight);
                int qmin = table_idx[5*ent_idx + 2], qmax = table_idx[5*ent_idx + 3], ench = table_idx[5*ent_idx + 4];
                if (ench) {
                    uint64_t ENCH_PACKED = 0b11001111100111101111111101110111111L; // which enchantments need a second call
                    int ench_idx = next_int(&seed, 37);
                    if ((ENCH_PACKED >> ench_idx) & 1) next_int(&seed, 1); // ignore result, for now
                }
                int qty = next_int_util(&seed, qmin, qmax);
                loot[table_idx[5*ent_idx]] += qty;
            }
            table_idx += 5*num_ent;
        }
        int match = 1;
        for (int i=1; i<d_distinct_items; i++) { // start at 1 (ignore empty)
            if (loot[i] != d_target[i]) match = 0;
        }
        if (match) printf("%ld\n", seed_start+dseed);
    }
}

void print_table(uint8_t* table) {
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


    printf("Launching kernel...\n");
    check_loot_collision<<<256, 256>>>(
        seed, 1<<30
    );

    cudaDeviceSynchronize();
}

int main() {
    int64_t seed = 6721027238469L;
    uint8_t table[107] = {2, 2, 4, 15, 1, 5, 1, 3, 0, 2, 15, 1, 5, 0, 3, 15, 2, 7, 0, 4, 15, 1, 3, 0, 5, 25, 4, 6, 0, 6, 25, 1, 3, 0, 7, 25, 3, 7, 0, 8, 20, 1, 1, 0, 9, 15, 1, 1, 0, 10, 10, 1, 1, 0, 11, 5, 1, 1, 0, 12, 20, 1, 1, 1, 13, 20, 1, 1, 0, 14, 2, 1, 1, 0, 0, 15, 1, 1, 0, 4, 4, 5, 5, 10, 1, 8, 0, 15, 10, 1, 8, 0, 7, 10, 1, 8, 0, 16, 10, 1, 8, 0, 17, 10, 1, 8, 0};
    uint8_t target[18] = {0, 0, 0, 0, 0, 17, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 3};
    int distinct_items = 18;
    if (distinct_items > MAX_DISTINCT_ITEMS) {
        printf("ERROR - too many items in loot table. Recompile with larger MAX_DISTINCT_ITEMS.\n");
        exit(1);
    }
    // todo: check for loot table too large
    h_table = table;
    h_target = target;
    h_distinct_items = distinct_items;

    print_table(table);
    gpu_init(seed-555);
//    check_loot_collision(seed, 1<<20);

    return 0;
}




