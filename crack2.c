#include <stdint.h>
#include <stdio.h>

#define LCG_MUL 0x5DEECE66DL
#define LCG_ADD 0xB
#define LCG_MSK 0xFFFFFFFFFFFFL

int64_t next(int64_t* seed, int bits) {
    *seed = (*seed * LCG_MUL + LCG_ADD) & LCG_MSK;
    return (uint64_t)*seed >> (48 - 31);
}

int32_t next_int(int64_t* seed, int32_t bound) {
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

int32_t next_int_util(int64_t* seed, int32_t min, int32_t max) {
    return min >= max ? min : min + next_int(seed, max-min+1);
}

int check_loot_collision(uint8_t* table, int64_t seed, uint8_t* target, uint8_t distinct_items) {
    int loot[256] = {0}; // todo - size smartly. needs only distinct_items length
    int num_pools = table[0];
    table += 1;
    for (int pool_idx=0; pool_idx<num_pools; pool_idx++) {
        int rmin = table[0], rmax = table[1];
        int num_ent = table[2];
        table += 3;
        int tot_weight = 0;
        for (int ent_idx=0; ent_idx<num_ent; ent_idx++) tot_weight += table[5*ent_idx + 1];
        int num_rolls = next_int_util(&seed, rmin, rmax);
        for (int roll_num=0; roll_num<num_rolls; roll_num++) {
            int cur_weight = 0, ent_idx=-1;
            int targ_weight = next_int_util(&seed, 1, tot_weight);
            do {cur_weight += table[5*(++ent_idx) + 1];} while (cur_weight < targ_weight);
            int qmin = table[5*ent_idx + 2], qmax = table[5*ent_idx + 3], ench = table[5*ent_idx + 4];
            if (ench) {
                uint64_t ENCH_PACKED = 0b11001111100111101111111101110111111L; // which enchantments need a second call
                int ench_idx = next_int(&seed, 37);
                if ((ENCH_PACKED >> ench_idx) & 1) next_int(&seed, 1); // ignore result, for now
            }
            int qty = next_int_util(&seed, qmin, qmax);
            loot[table[5*ent_idx]] += qty;
        }
        table += 5*num_ent;
    }
    for (int i=1; i<distinct_items; i++) { // start at 1 (ignore empty)
        if (loot[i] != target[i]) return 0;
    }
    return 1;
}
/*108852067794538
108852079326759
108852270193959
108852303847671
108852329813214
*/

int main() {
int64_t seed = 6721027238469;
uint8_t table[107] = {2, 2, 4, 15, 1, 5, 1, 3, 0, 2, 15, 1, 5, 0, 3, 15, 2, 7, 0, 4, 15, 1, 3, 0, 5, 25, 4, 6, 0, 6, 25, 1, 3, 0, 7, 25, 3, 7, 0, 8, 20, 1, 1, 0, 9, 15, 1, 1, 0, 10, 10, 1, 1, 0, 11, 5, 1, 1, 0, 12, 20, 1, 1, 1, 13, 20, 1, 1, 0, 14, 2, 1, 1, 0, 0, 15, 1, 1, 0, 4, 4, 5, 5, 10, 1, 8, 0, 15, 10, 1, 8, 0, 7, 10, 1, 8, 0, 16, 10, 1, 8, 0, 17, 10, 1, 8, 0};
uint8_t target[18] = {0, 0, 0, 0, 0, 17, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 3};
int distinct_items = 18;
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

    for (int ds=0; ds<(1<<20); ds++) {
        if (check_loot_collision(table, seed+ds, target, distinct_items)) {
            printf("%ld\n", seed+ds);
        }
    }
}