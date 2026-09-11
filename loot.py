from extract import get_chest_loot_table

def loottable_c(table_name, loot_item_counts):
    lookup = {'minecraft:empty':0, 'miss':1}
    table = get_chest_loot_table(table_name)
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
