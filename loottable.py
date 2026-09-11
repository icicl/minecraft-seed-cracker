def check_keys(obj, keys, prefix=''):
    assert type(obj) is dict, f"Expected dict for {obj}"
    assert len(obj) == len(keys), f"Expected {len(keys)} keys, got {len(obj)} for {obj}"
    assert all(prefix+key in obj for key in obj), f"Unexpected keys for object {obj}"


def get_range(obj, require_explicit_uniform=True):
    if type(obj) is int:
        return obj, obj
    elif type(obj) is dict:
        if require_explicit_uniform:
            check_keys(obj, ['min', 'max', 'type'])
            assert obj['type'] == 'minecraft:uniform'
        else:
            check_keys(obj, ['min', 'max'])
        rmin, rmax = obj['min'], obj['max']
        assert int(rmin) == rmin and int(rmax) == rmax
        return int(rmin), int(rmax)
    else:
        raise ValueError(f"Unknown range format: {obj}")


####### Functions #######
# 1: enchant_randomly - randomly pick enchant from all enchants (filtered to item type), then pick level (if applicable).
# 2: enchant_randomly - randomly pick enchant from the specified list of enchants, then pick level if applicable
# 3: set_damage - [presumably] pick random float f, lerp to range
# 4: exploration_map - buried treasure map
# 5: set_stew_effect - pick element from list, then set duration
def load_table(table):
    out = []
    check_keys(table, ['type', 'pools'])
    assert table['type'] == 'minecraft:chest'

    pools = table['pools']
    assert type(pools) is list
    for pool in pools:
        check_keys(pool, ['rolls', 'entries'])
        rolls = get_range(pool['rolls'])

        entries = pool['entries']
        assert type(entries) is list
        out.append([rolls, []])
        for entry in entries:
            entry_type = entry['type']
            if entry_type == 'minecraft:item':
                assert all(key in ['type', 'weight', 'functions', 'name'] for key in entry), f"Bad key for entry {entry}"
                weight = entry.get('weight', 1)
                assert type(weight) is int
                functions = entry.get('functions', [])
                assert type(functions) == list
                count = (1,1)
                funcs_out = []
                for function in functions:
                    func_name = function['function']
                    if func_name == 'minecraft:set_count':
                        check_keys(function, ['function', 'count'])
                        count = get_range(function['count'])
                    elif func_name == 'minecraft:enchant_randomly':
                        if 'enchantments' in function:
                            check_keys(function, ['function', 'enchantments'])
                            funcs_out.append(['enchant_filtered', function['enchantments']])
                        else:
                            check_keys(function, ['function'])
                            funcs_out.append(['enchant'])
                    elif func_name == 'minecraft:set_damage':
                        check_keys(function, ['function', 'damage'])
                        check_keys(function['damage'], ['min', 'max'])
                        dmin, dmax = function['damage']['min'], function['damage']['max']
                        assert type(dmin) is float and type(dmax) is float
                        funcs_out.append(['set_damage', (dmin,dmax)])
                    elif func_name == 'minecraft:enchant_with_levels':
                        check_keys(function, ['function', 'levels', 'treasure'])
                        levels = get_range(function['levels'])
                        assert function['treasure'] == True
                        funcs_out.append(['enchant_levels', levels])
                    elif func_name == 'minecraft:exploration_map':
                        check_keys(function, ['function', 'decoration', 'zoom', 'skip_existing'])
                        funcs_out.append(['map'])
                    elif func_name == 'minecraft:set_stew_effect':
                        check_keys(function, ['function', 'effects'])
                        funcs_out.append(['stew',[]])
                        for effect in function['effects']:
                            check_keys(effect, ['type', 'duration'])
                            dur = get_range(effect['duration'], require_explicit_uniform=False)
                            funcs_out[-1][-1].append([effect['type'],dur])
                    else:
                        print(f"Unknown function: {func_name}.")
                out[-1][-1].append([[entry['name'],count,funcs_out],weight])
            elif entry_type == 'minecraft:empty':
                assert all(key in ['type', 'weight'] for key in entry), f"Bad key for entry {entry}"
                weight = entry.get('weight', 1)
                assert type(weight) is int
                out[-1][-1].append([[None,(1,1),[]],weight])
            else:
                raise ValueError(f"Unknown type {entry_type} for entry {entry}")
    return out
