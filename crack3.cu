#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include <sys/time.h>

uint64_t time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t micros = (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
    return micros;
}

enum structures {
    DESERT_TEMPLE,
    BURIED_TREASURE
};

#define FLOAT_0_01_LIM (((uint64_t)(0.01f * (1 << 24)) + 1) << 24)

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
__device__ unsigned long long  d_match_counts[4];
__constant__ int64_t  d_spawn_checks[MAX_SPAWN_CHECKS*4];
__constant__ uint8_t d_num_spawn_checks;
__constant__ int64_t d_buried_treasure_float;

uint8_t* h_table;
uint8_t* h_target;
uint8_t  h_distinct_items;
uint8_t  h_popcnt;
uint32_t h_shuffle_order;
int32_t* h_feature_seed_info;
int64_t* h_spawn_checks;
uint8_t h_num_spawn_checks;
uint64_t h_buried_treasure_float;
uint64_t h_match_counts[4];

__device__ int64_t reverse_prng_call(int64_t rng_out, int64_t packed_info) {
    int64_t prev = (rng_out * 0xdfe05bcb1365 + 0x615c0e462aa9);
    prev ^= LCG_MUL;
    prev -= packed_info;
    return prev & LCG_MSK;
}

__device__ uint8_t can_spawn_desert_pyramid(int64_t seed_with_salt, int32_t mcx, int32_t mcz) {
    int64_t seed = (seed_with_salt ^ LCG_MUL) & LCG_MSK;
    return next_int(seed, 32 - 8) == mcx && next_int(seed, 32 - 8) == mcz;
}

__device__ uint8_t can_spawn_buried_treasure(int64_t seed_with_salt) {
    int64_t seed = (seed_with_salt ^ LCG_MUL) & LCG_MSK;
    return next(seed, 24) < 167773; // 0.01*(2^24) = 167772.2
}

__device__ uint8_t spawn_check(int64_t wseed, int64_t packed_info) {
    uint8_t structure_type = packed_info >> (48 + 2*6), mx = (packed_info >> (48 + 6)) & (0b111111), mz = (packed_info >> 48) & 0b111111;
    switch (structure_type) {
        case DESERT_TEMPLE:
            if (!can_spawn_desert_pyramid(wseed + packed_info, mx, mz)) return 0;
            break;
        case BURIED_TREASURE:
            if (!can_spawn_buried_treasure(wseed + packed_info)) return 0;
            break;
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


__global__ void check_loot_collision(uint32_t test_kernel) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ uint32_t s_table[MAX_LOOT_TABLE_SIZE/4];
    __shared__ uint8_t s_target[MAX_DISTINCT_ITEMS];
    __shared__ int64_t s_spawn_checks[MAX_SPAWN_CHECKS];
    for (int i=threadIdx.x; i<MAX_LOOT_TABLE_SIZE/4; i+=blockDim.x) s_table[i] = ((uint32_t*)d_table)[i];
    for (int i=threadIdx.x; i<d_distinct_items; i+=blockDim.x) s_target[i] = d_target[i];
    for (int i=threadIdx.x; i<d_num_spawn_checks; i+=blockDim.x) s_spawn_checks[i] = d_spawn_checks[4*i+3];
    __syncthreads();
    int8_t loot[MAX_DISTINCT_ITEMS];
    uint64_t spawn_ok_count = 0;

    const int32_t x = d_feature_seed_info[0], z = d_feature_seed_info[1], index = d_feature_seed_info[2], step = d_feature_seed_info[3];
    const uint64_t bt_reverse_float_accel = d_buried_treasure_float;
/*
    int64_t aa = 7577095897946LL;
    for (int64_t u = 17+24*tx; u < (1LL << 31); u += 24*stride) {
        for (int64_t l = 0; l < (1LL << 17); l += 1) {
            int64_t rout = (u << 17) | l;
            int64_t prev = (rout * 0xdfe05bcb1365 + 0x615c0e462aa9) & LCG_MSK;
            prev ^= LCG_MUL;
            prev -= aa;
            prev &= LCG_MSK;
            if (prev < 1000 && prev >= 500) printf("%ld\n", prev);
        }
    }
    return;
*/
    int64_t seed, wseed;

    // Buried treasure
    int64_t inner_lb, inner_ub, inner_stride;
    int64_t outer_lb, outer_ub, outer_stride;

    switch (bt_reverse_float_accel >> 48) {
        case DESERT_TEMPLE:
            inner_lb = 0, inner_ub = (1LL<<17), inner_stride = (test_kernel ? test_kernel : 1);
            outer_lb = (17LL + 24*tx)<<17, outer_ub = (1LL << 48), outer_stride = (24LL*stride)<<17;
            break;
        case BURIED_TREASURE:
            inner_lb = tx, inner_ub = FLOAT_0_01_LIM, inner_stride = (test_kernel ? test_kernel : 1)*stride;
            outer_lb = 0, outer_ub = 1, outer_stride = 1;
            break;
    }
    // Desert pyramid
//    int64_t inner_lb = 0, inner_ub = (1LL<<17), inner_stride = 1;
//    int64_t outer_lb = (17LL + 24*tx)<<17, outer_ub = (1LL << 48), outer_stride = (24LL*stride)<<17;
// 7577095897946LL

    atomicAdd(d_match_counts+0, ((outer_ub - outer_lb - 1) / outer_stride + 1) * ((inner_ub - inner_lb - 1) / inner_stride + 1));
    for (int64_t oi = outer_lb; oi < outer_ub; oi += outer_stride) {
        for (int64_t ii = inner_lb; ii < inner_ub; ii += inner_stride) {
            wseed = reverse_prng_call(oi + ii, bt_reverse_float_accel);
            uint8_t spawn_ok = 1;
            for (int j=0; j<d_num_spawn_checks && spawn_ok; j++) {
                spawn_ok &= spawn_check(wseed, s_spawn_checks[j]);
            }
            if (!spawn_ok) continue;
            spawn_ok_count++;
            
            for (int calls=0; calls<4; calls++) { // DES TEMPLE TODO parameterize (use x/z to get call#)
                int64_t feat_seed = get_lcg_feature_seed(wseed, x, z, index, step, calls) & LCG_MSK;
                seed = feat_seed ^ LCG_MUL;

                for (int i=1;i<d_distinct_items; i++) loot[i] = s_target[i];
                loot[0] = INT8_MAX; // don't want to break if 'empty' (id=0) rolled
                uint32_t* table_idx = s_table; // aligned to 4-bytes:    [num_pools | pad | pad | pad] { [rmin | rmax | totweight(2-byte) ] { [item_id | qmin | qmax | ench] } }
                int num_pools = table_idx[0] & 0xFF;
                table_idx += 1;
                int match = 1;
                int pool_idx = num_pools;
                while ((pool_idx--) && match) {
                    uint32_t pool_stats = table_idx[0];
                    int rmin = pool_stats & 0xFF, rmax = (pool_stats >> 8) & 0xFF;
                    int tot_weight = pool_stats >> 16;
                    int num_rolls = next_int_util(seed, rmin, rmax);
                    for (int roll_num=0; roll_num<num_rolls; roll_num++) {
                        int ent_idx = next_int_util(seed, 1, tot_weight);
                        uint32_t entry_stats = table_idx[ent_idx]; // table is a LUT - each possible chosen cum. weight has an entry for the corresponding item. This info is calculated in the invoking python
                        int qmin = (entry_stats >> 8) & 0xFF, qmax = (entry_stats >> 16) & 0xFF, ench = (entry_stats >> 24);
                        if (ench) { // TODO -- acount for tool/armor filtering
                            int ench_idx = next_int(seed, 37);
                            if ((ENCH_PACKED >> ench_idx) & 1) next_int(seed, 1); // ignore result (already have enough info w/o it)
                        }
                        int qty = next_int_util(seed, qmin, qmax);
                        if ((loot[entry_stats & 0xFF] -= qty) < 0) {
                            match = 0;
                            break;
                        };
                    }
                    table_idx += (1 + tot_weight); // jump ahead by 1 entry (num. roll and tot weight info), plus the tot_weight LUT entries
                }
                for (int i=1; (i<d_distinct_items)&&match; i++) { // start at 1 (ignore minecraft:empty at index 0)
                    if (loot[i] != 0) match = 0;
                }

                if (match && !test_kernel) {
                    atomicAdd(d_match_counts+2, 1);
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
                        atomicAdd(d_match_counts+3, 1);
                    }
                }
            }
        }
    }
    atomicAdd(d_match_counts+1, spawn_ok_count);
}

void gpu_init() {
    printf("Copying data to GPU...\n");
    cudaMemcpyToSymbol(d_table, h_table, MAX_LOOT_TABLE_SIZE*sizeof(uint8_t));
    cudaMemcpyToSymbol(d_target, h_target, MAX_DISTINCT_ITEMS*sizeof(uint8_t));
    cudaMemcpyToSymbol(d_distinct_items, &h_distinct_items, sizeof(uint8_t));
    cudaMemcpyToSymbol(d_popcnt, &h_popcnt, sizeof(uint8_t));
    cudaMemcpyToSymbol(d_shuffle_order, &h_shuffle_order, sizeof(uint32_t));
    cudaMemcpyToSymbol(d_num_spawn_checks, &h_num_spawn_checks, sizeof(uint8_t));
    cudaMemcpyToSymbol(d_spawn_checks, h_spawn_checks, 4*sizeof(int64_t)*MAX_SPAWN_CHECKS);
    cudaMemcpyToSymbol(d_feature_seed_info, h_feature_seed_info, 4*sizeof(int32_t));
    cudaMemcpyToSymbol(d_buried_treasure_float, &h_buried_treasure_float, sizeof(uint64_t));
    cudaMemset(&d_match_counts, 0, 4*sizeof(unsigned long long));
//    cudaMemset(&d_correct_count, 0, sizeof(uint32_t));
//    cudaMemset(&d_spawn_ok_count, 0, sizeof(uint64_t));

    printf("Launching test kernel...\n");

    uint64_t timer;
    uint32_t test_kernel_size = 16384;
    do {
        test_kernel_size /= 2;
        timer = -time_us();
        check_loot_collision<<<1024, 256>>>(test_kernel_size);
        cudaDeviceSynchronize();
        timer += time_us();
    } while (timer < 200000);
    printf("Test kernel finished in %.1fms\n",(float)timer / 1000);
    printf("Estimated time to check all seeds: %.1fs\n\n", (float)timer / 1000000 * test_kernel_size);
    printf("Launching full kernel...\n");

    check_loot_collision<<<1024, 256>>>(0);
    cudaDeviceSynchronize();



    cudaMemcpyFromSymbol(&h_match_counts, d_match_counts, 4*sizeof(unsigned long long*));
    printf("Checked %lu seeds. \n%lu passed structure spawn check.\nFFiltered to %lu matches using item quantities in loot.\nFurther filtered to %lu matches using the shuffle of empty slots.\n", h_match_counts[0], h_match_counts[1], h_match_counts[2], h_match_counts[3]);
}

int main() {
uint32_t shuffle_order = 0b110011110101101011011111010;
uint8_t table[1140] = {2, 0, 0, 0, 2, 4, 232, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 1, 5, 0, 1, 2, 7, 0, 1, 2, 7, 0, 1, 2, 7, 0, 1, 2, 7, 0, 1, 2, 7, 0, 1, 2, 7, 0, 1, 2, 7, 0, 1, 2, 7, 0, 1, 2, 7, 0, 1, 2, 7, 0, 1, 2, 7, 0, 1, 2, 7, 0, 1, 2, 7, 0, 1, 2, 7, 0, 1, 2, 7, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 1, 3, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 3, 3, 7, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 4, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 5, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 4, 4, 50, 0, 2, 1, 8, 0, 2, 1, 8, 0, 2, 1, 8, 0, 2, 1, 8, 0, 2, 1, 8, 0, 2, 1, 8, 0, 2, 1, 8, 0, 2, 1, 8, 0, 2, 1, 8, 0, 2, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 3, 1, 8, 0, 3, 1, 8, 0, 3, 1, 8, 0, 3, 1, 8, 0, 3, 1, 8, 0, 3, 1, 8, 0, 3, 1, 8, 0, 3, 1, 8, 0, 3, 1, 8, 0, 3, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 1, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0, 6, 1, 8, 0};
uint8_t target[7] = {0, 0, 16, 8, 1, 1, 2};
int distinct_items = 7;
int32_t feature_seed_info[4] = {7452, 10970, 3, 4}; // x, z, index, step
int64_t buried_treasure_float = 7577095897946LL;

int64_t spawns_checks[12] = {7984, 10480, DESERT_TEMPLE, 0, 7450, 10968, DESERT_TEMPLE};
int num_spawn_checks = 2;
//uint64_t buried_treasure_float = 1153169326606791148LL; // 75998109126254LL
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
    if (buried_treasure_float == 0) {
        printf("WARNING - you are running the cracker without a buried treasure position. This will increase the runtime by ~100x.\n");
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
    h_buried_treasure_float = buried_treasure_float;


    gpu_init();

    return 0;
}




