from functools import cache
from math import comb, log2
from extract import load_all_tables

def likelihood(table, items):
    lookup = {}
    inv = [item for item in items]
    for item in items: lookup[item] = len(lookup)
    qtys = [items[item] for item in inv]
    pools = []
    for roll_range,entries in table:
        tot_weight = sum(entry[1] for entry in entries)
        miss_weight = 0
        pools.append([roll_range,[None]*len(items)])
        for entry in entries:
            item = entry[0][0]
            if item in items:
                pools[-1][1][lookup[item]] = (entry[0][1], entry[1]/tot_weight)
            elif item is None:
                miss_weight += entry[1]
            else:
                pass
        pools[-1].append(miss_weight/tot_weight)

    @cache
    def prob(remaining, cur_pool, rem_pool_draws):
        if min(remaining) < 0: return 0
        if cur_pool == len(pools): return 1 if sum(remaining)==0 else 0
        result = 0
        if rem_pool_draws == 0:
            cur_pool += 1
            if cur_pool == len(pools): return 1 if sum(remaining)==0 else 0
            (rmin,rmax),entries,oth_prob = pools[cur_pool]
            for rolls in range(rmin, rmax+1):
                result += (1 / (rmax-rmin+1))*prob(remaining, cur_pool, rolls)
                
        else:
            (rmin,rmax),entries,oth_prob = pools[cur_pool]
            for iidx,item in enumerate(entries):
                if item is not None:
                    (qmin,qmax),p = item
                    for qty in range(qmin,qmax+1):
                        nxt = list(remaining)
                        nxt[iidx] -= qty
                        result += (1 / (qmax-qmin+1))*prob(tuple(nxt), cur_pool, rem_pool_draws-1)*p
            result += oth_prob * prob(remaining, cur_pool, rem_pool_draws-1)
        return result
    return prob(tuple(qtys), -1, 0)

def entropy(table, items, chest):
    p1 = likelihood(load_all_tables()[table], items) # item quantities
    p2 = 1 / comb(27, sum(e[0] is None for e in chest)) # empty/occupied positions
    p3 = 1 / {'desert_pyramid':576, 'buried_treasure':100, 'ruined_portal':625}.get(table, 1) # structure spawn chance
    p4 = 1 / {'desert_pyramid':24, 'buried_treasure':100, 'ruined_portal':25}.get(table, 1) # fast PRNG reversal sparsity
    def h(p):
        if p == 0: return float('inf')
        if p == 1: return 0.0
        return -log2(p)
    h1 = h(p1)
    h2 = h(p2)
    h3 = h(p3)
    h4 = h(p4)
    return h1,h2,h3,h4