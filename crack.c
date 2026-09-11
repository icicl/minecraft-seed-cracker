/*def shuffle(rng, arr):
    for i in range(len(arr), 1, -1):
        j = rng.next_int(i)
        arr[i-1],arr[j] = arr[j],arr[i-1]

indices = list(range(container_size))
shuffle(rng, indices)
...
for idx,stack in zip(indices[::-1],stacks): output[idx] = stack

class JavaRandom:
    def __init__(self,seed):
        self.seed=(seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
    def next(self,bits):
        self.seed=(self.seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return (self.seed >> (48 - bits))
    def next_int(self,bound):#fix sign
        if ((bound & -bound) == bound):# i.e., bound is a power of 2
            return ((bound * self.next(31)) >> 31);    
        bits=self.next(31)
        val=bits%bound
        while bits-val+(bound-1)<0:
            bits=self.next(31)
            val=bits%bound
        return val
    def prev(self):
        self.seed=(self.seed*0xdfe05bcb1365+0x615c0e462aa9)&((1<<48)-1)*/
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

int main() {
    uint32_t target = 0b000111111111111110101110000;
    int popcnt = 0;
    for (int i=0; i<27; i++) popcnt += ((target >> i)&1);

    int64_t start = 239681203944955;
    int64_t amount = 10000000;

    for (int64_t ds=0; ds<amount; ds++) {
        int64_t seed = start+ds;
        uint8_t ind[27];
        for (int i=0; i<27; i++) {
            ind[i] = i;
        }
        for (int i=27; i>1; i--) {
            int j = next_int(&seed, i);
            uint8_t tmp = ind[j];
            ind[j] = ind[i-1];
            ind[i-1] = tmp;
        }
        int8_t correct = 1;
        for (int i=27-popcnt; i<27; i++) {
            if (((target >> (26-ind[i])) & 1) == 0) {
                correct = 0;
                break;
            }
        }
        if (correct) {
            printf("%ld\n", start+ds);
        }
    }
}