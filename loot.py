import json
from extract import get_chest_loot_table
from enchants import get_random_enchant
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
    def next_long(self):
        return (self.next(32) << 32) + self.next(32)
    def next_long_(self):
        _ = (self.next(32)<<32)
        __=self.next(32)
        if _>=(1<<63):_-=(1<<64)
        if __>=(1<<31):_-=(1<<32)
        return _+__
    def next_bool(self):
        return self.next(1)!=0
    def next_float(self):
        return self.next(24)/(1<<24)
    def prev(self):
        self.seed=(self.seed*0xdfe05bcb1365+0x615c0e462aa9)&((1<<48)-1)

def reverse_pool(rolls, entries, results):
    states = [(results.copy(),[])]
    states = [({},[])]
    for _ in range(rolls):
        possible = []
        if len(entries) == 1:
            entry = entries[0]
            possible.append((entry,[]))
        else:
            tot_weight = 0
            for entry in entries: tot_weight += entry.get('weight', 1)
            cum_weight = 0
            for entry in entries:
                w = entry.get('weight', 1)
                cum_weight += w
                possible.append((entry,[(tot_weight,cum_weight-w,cum_weight)]))
        possible_ = []
        for entry,rhist in possible:
            if 'functions' in entry:
                assert len(entry['functions']) == 1
                if entry['functions'][0]['function'] == 'minecraft:enchant_randomly':
                    possible_.append(((entry['name'],1),rhist+[(1,0,1)]))
                    possible_.append(((entry['name'],1),rhist+[(1,0,1),(1,0,1)])) # add both (may rarely give false positive)
                elif entry['functions'][0]['function'] == 'minecraft:set_count':
                    assert entry['functions'][0]['count']['type'] == 'minecraft:uniform'
                    qmin = entry['functions'][0]['count']['min']
                    qmax = entry['functions'][0]['count']['max']
                    assert int(qmin) == qmin and int(qmax) == qmax
                    qmin = int(qmin)
                    qmax = int(qmax)
                    for qty in range(qmax - qmin + 1):
                        possible_.append(((entry['name'],qty+qmin),rhist+[(qmax-qmin+1,qty,qty+1)]))
                else:
                    raise ValueError
            else:
                if entry['type'] == 'minecraft:empty':
                    possible_.append(((None,0),rhist))
                else:
                    possible_.append(((entry['name'],1),rhist))
        states_ = []
        for res,rhist in states:
            for (item,qty),shist in possible_:
                if qty + res.get(item,0) <= results.get(item,0):
                    states_.append((res.copy(),rhist+shist))
                    if qty > 0:
                        states_[-1][0][item] = states_[-1][0].get(item,0)+qty
        states = states_
    def key(dct):
        return str(sorted(dct.items()))
    out = []
    ind = {}
    for dct,hst in states:
        k = key(dct)
        if k not in ind:
            ind[k] = len(ind)
            out.append((dct,[]))
        out[ind[k]][-1].append(hst)
    return out

from itertools import product
def reverse_rng_calls(table, results):
    assert table['type'] == 'minecraft:chest'
    all_pulls = []

    pools = table['pools']
    for pool in pools:
            pulls = []
            rolls = pool['rolls']
            if type(rolls) is int:
                    for res_p,his_p in reverse_pool(rolls,pool['entries'],results):
                        pulls.append((res_p,[his_p]))
            elif type(rolls) is dict:
                rmin = rolls['min']
                rmax = rolls['max']
                assert rolls['type'] == 'minecraft:uniform'
                assert int(rmin) == rmin and int(rmax) == rmax
                rmin = int(rmin)
                rmax = int(rmax)
                for rng in range(rmax-rmin+1):
                    for res_p,his_p in reverse_pool(rmin+rng,pool['entries'],results):
                        pulls.append((res_p,[[(rmax-rmin+1,rng,rng+1)]]+[his_p]))
            else:
                raise ValueError
            all_pulls.append(pulls)
    for dcts in product(*[range(len(p)) for p in all_pulls]):
        if all(sum(all_pulls[i][j][0].get(item,0) for i,j in enumerate(dcts)) == r_qty for item,r_qty in results.items()):
            for i,j in enumerate(dcts):
                print(all_pulls[i][j])
                print(list(map(len,all_pulls[i][j][1])))
            print()




def process_loot_pool(pool, rng):
    rolls = pool['rolls']
    if type(rolls) is int:
        pass
    elif type(rolls) is dict:
        rmin = rolls['min']
        rmax = rolls['max']
        assert rolls['type'] == 'minecraft:uniform'
        assert int(rmin) == rmin and int(rmax) == rmax
        rmin = int(rmin)
        rmax = int(rmax)
        rolls = rng.next_int(rmax - rmin + 1) + rmin
    else:
        raise ValueError
    entries = pool['entries']
    for _ in range(rolls):
        if len(entries) == 1:
            entry = entries[0]
        else:
            tot_weight = 0
            for entry in entries: tot_weight += entry.get('weight', 1)
            pick = rng.next_int(tot_weight)
            tot_weight = 0
            for entry in entries:
                tot_weight += entry.get('weight', 1)
                if tot_weight > pick: break
        if 'functions' in entry:
            assert len(entry['functions']) == 1
            function = entry['functions'][0]['function']
            if function == 'minecraft:enchant_randomly':
                enchant,max_level = get_random_enchant(entry['name'], rng)
                level = r_int(rng, 1, max_level)
                print(enchant, level)
                qty = 1
            elif function == 'minecraft:set_stew_effect':
                effects = entry['functions'][0]['effects']
                effect = effects[r_int(rng, 0, len(effects)-1)]
                emin = effect['duration']['min']
                emax = effect['duration']['max']
                assert int(emin) == emin and int(emax) == emax
                duration = r_int(rng, int(emin), int(emax))
                print(f"Stew {effect['type']}. Duration {duration}s.")
                qty = 1
            elif function == 'minecraft:set_count':
                assert entry['functions'][0]['count']['type'] == 'minecraft:uniform'
                qmin = entry['functions'][0]['count']['min']
                qmax = entry['functions'][0]['count']['max']
                assert int(qmin) == qmin and int(qmax) == qmax
                qmin = int(qmin)
                qmax = int(qmax)
                qty = qmin + rng.next_int(qmax - qmin + 1)
            elif function == 'minecraft:exploration_map':
                qty = 1
            else:
                print("Bad Entry Function")
                print(entry)
                raise ValueError
        else:
            qty = 1
        if entry['type'] == 'minecraft:empty':
            yield (None, 0)
        else:
            yield (entry['name'], qty)

def r_int(rng, lower, upper): # I believe the Mth nextint will only made a random call if needed
    if lower >= upper: return lower
    return rng.next_int(upper - lower + 1) + lower

def shuffle(rng, arr):
    for i in range(len(arr), 1, -1):
        j = rng.next_int(i)
        arr[i-1],arr[j] = arr[j],arr[i-1]


def get_loot_items(table, rng):
    assert table['type'] == 'minecraft:chest'
    pools = table['pools']

    stacks = []
    for pool in pools:
        for item in process_loot_pool(pool, rng):
            stacks.append(item)
    return stacks


def shuffle_and_split(stacks, rng, container_size=27):
    splittable, kept = [], []
    for s in stacks:
        count = s[1]
        if count <= 0:
            continue
        elif count > 1:
            splittable.append(s)
        else:
            kept.append(s)
    stacks = kept

    indices = list(range(container_size))
    shuffle(rng, indices)

    while container_size - len(stacks) - len(splittable) > 0 and len(splittable):
        idx = r_int(rng, 0, len(splittable) - 1)
        stack = splittable.pop(idx)

        name, count = stack

        split = r_int(rng, 1, count // 2)

        a = (name, count - split)
        b = (name, split)
        for s in (a,b):
            if s[1] > 1 and rng.next_bool():
                splittable.append(s)
            else:
                stacks.append(s)
    stacks.extend(splittable)
    shuffle(rng, stacks)
    output = [None]*27
    for idx,stack in zip(indices[::-1],stacks): output[idx] = stack
    return output

def loottable_c(table, loot_item_counts):
    lookup = {'minecraft:empty':0, 'miss':1}
    table = get_chest_loot_table(table)
    assert table['type'] == 'minecraft:chest'
    pools = table['pools']
    output = [len(pools),0,0,0] #     #pools < rmin rmax #entries totweight [ name weight qmin qmax ench ] >
                          #     #pools pad pad pad < rmin rmax #entries totweight [ name qmin qmax ench ] > 
    for pool in pools: #                                                ^ for each possible targ weight
        rolls = pool['rolls']
        if type(rolls) is int:
            output.extend([rolls, rolls])
        elif type(rolls) is dict:
            rmin = rolls['min']
            rmax = rolls['max']
            assert rolls['type'] == 'minecraft:uniform'
            assert int(rmin) == rmin and int(rmax) == rmax
            output.extend([int(rmin), int(rmax)])
        else:
            raise ValueError
        entries = pool['entries']
        tot_weight = sum(entry.get('weight', 1) for entry in entries)
        output.append(tot_weight&0xFF)
        output.append(tot_weight>>8)
        for entry in entries:
            ## TODO: account for enchantment filtering
            name = 'minecraft:empty' if entry['type'] == 'minecraft:empty' else (entry['name'] if entry['name'] in loot_item_counts else 'miss')
            if name not in lookup: lookup[name] = len(lookup)
            item_id = lookup[name]
            if 'functions' in entry:
                assert len(entry['functions']) == 1
                if entry['functions'][0]['function'] == 'minecraft:enchant_randomly':
                    entry_stats = [item_id, 1, 1, 1]
                elif entry['functions'][0]['function'] == 'minecraft:set_count':
                    assert entry['functions'][0]['count']['type'] == 'minecraft:uniform'
                    qmin = entry['functions'][0]['count']['min']
                    qmax = entry['functions'][0]['count']['max']
                    assert int(qmin) == qmin and int(qmax) == qmax
                    entry_stats = [item_id, int(qmin), int(qmax), 0]
                else:
                    raise ValueError
            else:
                entry_stats = [item_id, 1, 1, 0]
            output.extend(entry_stats*entry.get('weight', 1))
    assert max(output) < 256
    return output, lookup
