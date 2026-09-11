#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

enum structures {
    DESERT_TEMPLE,
    BURIED_TREASURE,
    RUINED_PORTAL,
};

#define FLOAT_0_01_LIM (((uint64_t)(0.01f * (1 << 24)) + 1) << 24)

#define LCG_MUL 0x5DEECE66DL
#define LCG_ADD 0xB
#define LCG_MSK 0xFFFFFFFFFFFFL

#define MAX_DISTINCT_ITEMS 16
#define MAX_LOOT_TABLE_SIZE 2048
#define MAX_SPAWN_CHECKS 8
#define MAX_DISTINCT_ENCHANTABLE_TOOLS 80


uint64_t time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t micros = (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
    return micros;
}


__constant__ uint8_t d_table[MAX_LOOT_TABLE_SIZE];
__constant__ uint8_t d_target[MAX_DISTINCT_ITEMS];
__constant__ uint8_t d_distinct_items;
__constant__ uint8_t d_popcnt;
__constant__ uint32_t d_shuffle_order;
__constant__ int32_t d_feature_seed_info[4];
__device__ unsigned long long  d_match_counts[5];
__constant__ int64_t  d_spawn_checks[MAX_SPAWN_CHECKS];
__constant__ uint8_t d_num_spawn_checks;
__constant__ int64_t d_prng_first_call_salt;
__constant__ uint16_t d_split_stacks[27];
__constant__ uint64_t d_enchant_callcounts[MAX_DISTINCT_ENCHANTABLE_TOOLS];


/* PRNG LOGIC */
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

__device__ int64_t reverse_prng_call(int64_t rng_out, int64_t packed_info) {
    int64_t prev = (rng_out * 0xdfe05bcb1365 + 0x615c0e462aa9);
    prev ^= LCG_MUL;
    prev -= packed_info;
    return prev & LCG_MSK;
}
/* END PRNG FUNCTIONS */


/* WORLD GEN FUNCTIONS */
__device__ uint8_t can_spawn_grid_based(int64_t seed_with_salt, int32_t mcx, int32_t mcz, uint8_t modulus) {
    int64_t seed = (seed_with_salt ^ LCG_MUL) & LCG_MSK;
    return next_int(seed, modulus) == mcx && next_int(seed, modulus) == mcz;
}

__device__ uint8_t can_spawn_prob_based(int64_t seed_with_salt) {
    int64_t seed = (seed_with_salt ^ LCG_MUL) & LCG_MSK;
    return next(seed, 24) < 167773; // 0.01*(2^24) = 167772.2
}

__device__ uint8_t spawn_check(int64_t wseed, int64_t packed_info) {
    uint8_t structure_type = packed_info >> (48 + 2*6), mx = (packed_info >> (48 + 6)) & (0b111111), mz = (packed_info >> 48) & 0b111111;
    switch (structure_type) {
        case DESERT_TEMPLE:
            return can_spawn_grid_based(wseed + packed_info, mx, mz, 24);
            break;
        case BURIED_TREASURE:
            return can_spawn_prob_based(wseed + packed_info);
            break;
        case RUINED_PORTAL:
            return can_spawn_grid_based(wseed + packed_info, mx, mz, 25);
            break;
    }
    return 0;
}

__device__ int64_t get_lcg_feature_seed(int64_t world_seed, int32_t x, int32_t z, int index, int step, int calls) {
    int64_t seed = (world_seed ^ LCG_MUL) & LCG_MSK;

    int64_t a = next_long(seed) | 1, b = next_long(seed) | 1;
    int64_t dec_seed = (x*a + z*b ^ world_seed) & LCG_MSK;
    int64_t feature_seed = (dec_seed + index + 10000*step) & LCG_MSK;
    feature_seed ^= LCG_MUL;
    for (int _=0; _<calls; _++) next_long(feature_seed);
    return next_long(feature_seed); 
}
/* END WORLD GEN FUNCTIONS */


__global__ void check_loot_collision(uint32_t num_dispatches, uint32_t dispatch_idx) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ uint32_t s_table[MAX_LOOT_TABLE_SIZE/4];
    __shared__ uint8_t s_target[MAX_DISTINCT_ITEMS];
    __shared__ int64_t s_spawn_checks[MAX_SPAWN_CHECKS];
    for (int i=threadIdx.x; i<MAX_LOOT_TABLE_SIZE/4; i+=blockDim.x) s_table[i] = ((uint32_t*)d_table)[i];
    for (int i=threadIdx.x; i<d_distinct_items; i+=blockDim.x) s_target[i] = d_target[i];
    for (int i=threadIdx.x; i<d_num_spawn_checks; i+=blockDim.x) s_spawn_checks[i] = d_spawn_checks[i];
    __syncthreads();
    int8_t loot[MAX_DISTINCT_ITEMS];
    uint64_t spawn_ok_count = 0;

    const int32_t x = d_feature_seed_info[0] & 0xFFFFFFF0, z = d_feature_seed_info[1] & 0xFFFFFFF0, index = d_feature_seed_info[2], step = d_feature_seed_info[3];
    const uint64_t prng_first_call_salt = d_prng_first_call_salt;

    __shared__ uint16_t stacks_data[27*256];
    uint16_t* stacks = stacks_data + 27*(threadIdx.x);
    uint8_t stacks_idx;

    int64_t seed, wseed;

    int64_t inner_lb, inner_ub, inner_stride;
    int64_t outer_lb, outer_ub, outer_stride;

    switch (prng_first_call_salt >> 56) {
        case DESERT_TEMPLE:
            inner_lb = dispatch_idx, inner_ub = (1LL<<17), inner_stride = num_dispatches;
            outer_lb = (((prng_first_call_salt >> 48) & 0xFF) + 24*tx)<<17, outer_ub = (1LL << 48), outer_stride = (24LL*stride)<<17;
            break;
        case BURIED_TREASURE:
            inner_lb = tx + dispatch_idx*stride, inner_ub = FLOAT_0_01_LIM, inner_stride = num_dispatches*stride;
            outer_lb = 0, outer_ub = 1, outer_stride = 1;
            break;
        case RUINED_PORTAL:
            inner_lb = dispatch_idx, inner_ub = (1LL<<17), inner_stride = num_dispatches;
            outer_lb = (((prng_first_call_salt >> 48) & 0xFF) + 25*tx)<<17, outer_ub = (1LL << 48), outer_stride = (25LL*stride)<<17;
            break;
    }

    atomicAdd(d_match_counts+0, ((outer_ub - outer_lb - 1) / outer_stride + 1) * ((inner_ub - inner_lb - 1) / inner_stride + 1));
    for (int64_t oi = outer_lb; oi < outer_ub; oi += outer_stride) {
        for (int64_t ii = inner_lb; ii < inner_ub; ii += inner_stride) {
            wseed = reverse_prng_call(oi + ii, prng_first_call_salt);
            uint8_t spawn_ok = 1;
            for (int j=0; j<d_num_spawn_checks && spawn_ok; j++) {
                spawn_ok &= spawn_check(wseed, s_spawn_checks[j]);
            }
            if (!spawn_ok) continue;
            spawn_ok_count++;
            
            for (int calls=0; calls<4; calls++) { // DES TEMPLE TODO parameterize (use x/z to get call#)
                int match = 1;
                for (uint8_t save_to_smem=0; save_to_smem<2; save_to_smem++) { // it is faster to re-run the loot calculation for the rare case when we get a hit, rather than saving the stack sizes to shared mem every time
                    int64_t feat_seed = get_lcg_feature_seed(wseed, x, z, index, step, calls) & LCG_MSK;
                    seed = feat_seed ^ LCG_MUL;

                    for (int i=1;i<d_distinct_items; i++) loot[i] = s_target[i];
                    loot[0] = INT8_MAX; // don't want to break if 'empty' (id=0) rolled
                    stacks_idx = 0;
                    uint32_t* table_idx = s_table; // aligned to 4-bytes:    [num_pools | pad | pad | pad] { [rmin | rmax | totweight(2-byte) ] { [item_id | qmin | qmax | ench] } }
                    int num_pools = table_idx[0] & 0xFF;
                    table_idx += 1;
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
                            if (ench) {
                                uint64_t ench_call_info = d_enchant_callcounts[ench];
                                uint8_t valid_num_tools = ench_call_info >> 56;
                                int ench_idx = next_int(seed, valid_num_tools);
                                if ((ench_call_info >> ench_idx) & 1) next_int(seed, 1); // ignore result (already have enough info w/o it)
                            }
                            int qty = next_int_util(seed, qmin, qmax);
                            if ((loot[entry_stats & 0xFF] -= qty) < 0) {
                                match = 0;
                                break;
                            };
                            if (save_to_smem) stacks[stacks_idx++] = (entry_stats << 8) | qty;
                        }
                        table_idx += (1 + tot_weight); // jump ahead by 1 entry (num. roll and tot weight info), plus the tot_weight LUT entries
                    }
                    for (int i=1; (i<d_distinct_items)&&match; i++) { // start at 1 (ignore minecraft:empty at index 0)
                        if (loot[i] != 0) match = 0;
                    }

                    if (match) {
                        if (save_to_smem == 0) continue;
                        atomicAdd(d_match_counts+2, 1);
                        uint8_t ind[27];
                        for (int i=0; i<27; i++) ind[i] = i;
                        for (int i=27; i>1; i--) {
                            int j = next_int(seed, i);
                            uint8_t tmp = ind[j];
                            ind[j] = ind[i-1];
                            ind[i-1] = tmp;
                        }
                        for (int i=27-d_popcnt; i<27; i++) {
                            if (((d_shuffle_order >> ind[i]) & 1) == 0) match = 0;
                        }
                        if (match) {
                            atomicAdd(d_match_counts+3, 1);
                            uint8_t st_idx = 0, sp_idx = 0;
                            uint16_t split[27];
                            for (uint8_t s_idx=0; s_idx < stacks_idx; s_idx++) { // put splittable (qty > 1) stacks at the left, unsplittable at the right
                                uint16_t item = stacks[s_idx];
                                uint8_t qty = item & 0xFF;
                                if (qty <= 0 || (item >> 8) == 0) continue; // skip if empty
                                if (qty >= 2) stacks[st_idx++] = item;
                                else split[sp_idx++] = item;
                            }
                            while (st_idx > 0 && (st_idx + sp_idx) < 27) {
                                uint8_t pop_idx = next_int_util(seed, 0, --st_idx);
                                uint16_t item = stacks[pop_idx];
                                for (uint8_t s_idx=pop_idx; s_idx<st_idx; s_idx++) stacks[s_idx] = stacks[s_idx+1];
                                uint8_t qty = item & 0xFF;
                                item &= 0xFF00;
                                uint8_t split_qty = next_int_util(seed, 1, qty/2);
                                qty -= split_qty;
                                if (qty > 1 && (next(seed, 1) != 0)) stacks[st_idx++] = item | qty;
                                else split[sp_idx++] = item | qty;
                                if (split_qty > 1 && (next(seed,1) != 0)) stacks[st_idx++] = item | split_qty;
                                else split[sp_idx++] = item | split_qty;
                            }
                            for (uint8_t s_idx=0; s_idx < st_idx; s_idx++) split[sp_idx++] = stacks[s_idx];
                            for (int i=sp_idx; i>1; i--) {
                                int j = next_int(seed, i);
                                uint16_t tmp = split[j];
                                split[j] = split[i-1];
                                split[i-1] = tmp;
                            }
                            for (int i=0; i<sp_idx; i++) {
                                match &= (split[i] == d_split_stacks[ind[26-i]]);
                            }
                            if (match) {
                                printf("Found match for 48-bit loot table seed: \e[1;35m%ld\e[0m                                              \n", wseed);
                                if ((wseed >> 32) == 0) printf("  -> Possible world seed, if seed was entered as a string (hashed to int32): \e[1;33m%ld\e[0m\n", wseed);
                                uint64_t s2_47_16 = wseed & 0xFFFFFFFF;
                                int32_t s1_31_16 = wseed >> 32;
                                if (s2_47_16 >= (1LL<<31)) s1_31_16++;
                                for (int32_t s2_15_0=0; s2_15_0 < (1<<16); s2_15_0++) {
                                    uint64_t prev = (((s2_47_16 << 16) | s2_15_0)*0xdfe05bcb1365LL + 0x615c0e462aa9) & LCG_MSK;
                                    if (((prev >> 16) & 0xFFFF) == s1_31_16) {
                                        int64_t seed64 = (((prev >> 16) & 0xFFFFFFFF) << 32) + (int32_t)s2_47_16;
                                        printf("  -> Possible world seed, if world was randomly generated: \e[1;33m%ld\e[0m\n", seed64);
                                    }
                                }
                                atomicAdd(d_match_counts+4, 1);
                            }
                        }
                    }
                }
            }
        }
    }
    atomicAdd(d_match_counts+1, spawn_ok_count);
}




//nvcc -O3 -Xcompiler -fPIC -shared crack3.cu -o crack.so
extern "C" // preserve function name
void dispatch(
    uint32_t shuffle_order,
    uint8_t* table,
    uint8_t* target,
    int distinct_items,
    int32_t* feature_seed_info,
    int64_t prng_first_call_salt,
    int64_t* spawn_checks,
    int num_spawn_checks,
    uint16_t* split_stacks,
    uint64_t* enchant_callcounts
) {
    int popcnt = 0;
    for (int i=0; i<27; i++) popcnt += ((shuffle_order >> i)&1);
    if (distinct_items > MAX_DISTINCT_ITEMS) {
        printf("ERROR - too many items in loot table. Recompile with larger MAX_DISTINCT_ITEMS.\n");
        exit(1);
    }
    if (enchant_callcounts[0] > MAX_DISTINCT_ENCHANTABLE_TOOLS) {
        printf("ERROR: please increase MAX_DISTINCT_ENCHANTABLE_TOOLS and recompile.");
        exit(1);
    }

    // todo: check for loot table too large
    printf("Copying data to GPU...\n");
    cudaMemcpyToSymbol(d_table, table, MAX_LOOT_TABLE_SIZE*sizeof(uint8_t));
    cudaMemcpyToSymbol(d_target, target, MAX_DISTINCT_ITEMS*sizeof(uint8_t));
    cudaMemcpyToSymbol(d_distinct_items, &distinct_items, sizeof(uint8_t));
    cudaMemcpyToSymbol(d_popcnt, &popcnt, sizeof(uint8_t));
    cudaMemcpyToSymbol(d_shuffle_order, &shuffle_order, sizeof(uint32_t));
    cudaMemcpyToSymbol(d_num_spawn_checks, &num_spawn_checks, sizeof(uint8_t));
    cudaMemcpyToSymbol(d_spawn_checks, spawn_checks, sizeof(int64_t)*MAX_SPAWN_CHECKS);
    cudaMemcpyToSymbol(d_feature_seed_info, feature_seed_info, 4*sizeof(int32_t));
    cudaMemcpyToSymbol(d_prng_first_call_salt, &prng_first_call_salt, sizeof(uint64_t));
    cudaMemset(&d_match_counts, 0, 5*sizeof(unsigned long long));
    cudaMemcpyToSymbol(d_split_stacks, split_stacks, 27*sizeof(uint16_t));
    cudaMemcpyToSymbol(d_enchant_callcounts, enchant_callcounts, MAX_DISTINCT_ENCHANTABLE_TOOLS*sizeof(uint64_t));

    uint64_t timer;
    uint32_t test_kernel_size = 16384;
    printf("Launching test kernel...\n");
    do {
        test_kernel_size /= 2;
        timer = -time_us();
        check_loot_collision<<<1024, 256>>>(test_kernel_size, 0);
        cudaDeviceSynchronize();
        timer += time_us();
    } while (timer < 200000);

    printf("Test kernel finished in %.1fms\n",(float)timer / 1000);
    printf("Estimated time to check all seeds: \e[1;34m%.1fs\e[0m\n\n", (float)timer / 1000000 * test_kernel_size);

    printf("Launching full kernel...\n");
    timer = -time_us();
    uint16_t num_dispatches = 1024;
//    const uint8_t braile[8] = {0b01000111, 0b11000110, 0b11100100, 0b11110000, 0b10111000, 0b00111001, 0b00011011, 0b00001111};
    const char* braile[] = {"⡇", "⣆", "⣤", "⣰", "⢸", "⠹", "⠛", "⠏"};
    for (uint16_t dispatch_idx=0; dispatch_idx<num_dispatches; dispatch_idx++) {
        check_loot_collision<<<1024, 256>>>(num_dispatches, dispatch_idx);
        cudaDeviceSynchronize();
        printf("%s%4.1f%% of seeds processed. %.1fs elapsed. ETA: %.1fs       \r", braile[dispatch_idx%8], 100.0*(dispatch_idx+1)/num_dispatches, 0.000001*(time_us()+timer), (0.000001*(time_us()+timer))*(num_dispatches - dispatch_idx - 1)/(dispatch_idx + 1));
        fflush(stdout);
    }
    uint64_t h_match_counts[5];
    cudaMemcpyFromSymbol(&h_match_counts, d_match_counts, 5*sizeof(unsigned long long*));
    printf("                                                           \n");
    printf("Execution finished in \e[1;34m%4.1fs\e[0m.\n", 0.000001*(time_us()+timer));
    printf("Checked \e[1;37m%lu\e[0m seeds. \n\e[1;37m%lu\e[0m passed structure spawn check.\nFiltered to \e[1;37m%lu\e[0m matches using item quantities in loot.\nFurther filtered to \e[1;37m%lu\e[0m matches using the shuffle of empty slots.\nReduced to \e[1;37m%lu\e[0m final seeds using the exact split+shuffle of items.\n", h_match_counts[0], h_match_counts[1], h_match_counts[2], h_match_counts[3], h_match_counts[4]);
}
