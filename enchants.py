from functools import cache

from extract import load_all_tables


def get_enchant_rng_call_data_c():
    lookup = get_enchant_lookup()
    data = [len(lookup) + 1] + [0]*len(lookup)
    for (valid,call_twice,idx) in lookup.values():
        assert valid <= 56
        data[idx] = (valid << 56) | call_twice
    return data

def get_enchantable_item_id(item):
    return get_enchant_lookup()[item][2]

@cache
def get_enchant_lookup():
    items_in_loot_table_with_enchant = set()
    for table in load_all_tables().values():
        for rolls,pool in table:
            for (item, qty, funcs), w in pool:
                if funcs:
                    if any('enchant' in f[0] for f in funcs):
                        items_in_loot_table_with_enchant.add(item)

    lookup = {}
    for item in sorted(items_in_loot_table_with_enchant):
        yes,no = [],[]
        call_twice = 0
        valid = 0
        for enchant in ALL_ENCHANTS:
            if can_enchant(enchant, item):
                call_twice = call_twice | ((ENCHANT_LEVELS[enchant] != 1) << valid)
                valid += 1
        lookup[item] = (valid, call_twice, len(lookup)+1)
    return lookup


def can_enchant(enchant, item_id):
    if "book" in item_id:
        return True
        
    # BREAKABLE
    if enchant in ["unbreaking", "mending", "vanishing_curse"]:
        return True

    # ARMOR
    is_armor = any(x in item_id for x in ["helmet", "chestplate", "leggings", "boots"])
    if enchant in ["protection", "fire_protection", "blast_protection", "projectile_protection"]:
        return is_armor
    if enchant == "thorns":
        return is_armor  # Overridden in Java! Base category is CHEST, but applies to all.
    if enchant == "binding_curse":
        return is_armor or "elytra" in item_id or "pumpkin" in item_id
        
    # SPECIFIC ARMOR SLOTS
    if enchant in ["respiration", "aqua_affinity"]:
        return "helmet" in item_id
    if enchant in ["feather_falling", "depth_strider", "frost_walker"]:
        return "boots" in item_id

    # WEAPONS & DIGGERS
    if enchant in ["sharpness", "smite", "bane_of_arthropods"]:
        return "sword" in item_id or "axe" in item_id  # Overridden in Java!
    if enchant in ["knockback", "fire_aspect", "looting", "sweeping"]:
        return "sword" in item_id  # Axes do NOT get these via enchant_randomly.

    # DIGGERS
    if enchant in ["efficiency", "silk_touch", "fortune"]:
        return any(x in item_id for x in ["pickaxe", "shovel", "axe", "hoe"])

    # RANGED & MISC
    if enchant in ["power", "punch", "flame", "infinity"]:
        return "bow" in item_id and "crossbow" not in item_id
    if enchant in ["multishot", "quick_charge", "piercing"]:
        return "crossbow" in item_id
    if enchant in ["luck_of_the_sea", "lure"]:
        return "fishing_rod" in item_id
    if enchant in ["loyalty", "impaling", "riptide", "channeling"]:
        return "trident" in item_id

    return False


ALL_ENCHANTS = [
    "protection", "fire_protection", "feather_falling", "blast_protection",
    "projectile_protection", "respiration", "aqua_affinity", "thorns",
    "depth_strider", "frost_walker", "binding_curse", "sharpness", "smite",
    "bane_of_arthropods", "knockback", "fire_aspect", "looting", "sweeping",
    "efficiency", "silk_touch", "unbreaking", "fortune", "power", "punch",
    "flame", "infinity", "luck_of_the_sea", "lure", "loyalty", "impaling",
    "riptide", "channeling", "multishot", "quick_charge", "piercing",
    "mending", "vanishing_curse"
]

ENCHANT_LEVELS = {
    "protection": 4, "fire_protection": 4, "feather_falling": 4, "blast_protection": 4,
    "projectile_protection": 4, "respiration": 3, "aqua_affinity": 1, "thorns": 3,
    "depth_strider": 3, "frost_walker": 2, "binding_curse": 1, "sharpness": 5, "smite": 5,
    "bane_of_arthropods": 5, "knockback": 2, "fire_aspect": 2, "looting": 3, "sweeping": 3,
    "efficiency": 5, "silk_touch": 1, "unbreaking": 3, "fortune": 3, "power": 5, "punch": 2,
    "flame": 1, "infinity": 1, "luck_of_the_sea": 3, "lure": 3, "loyalty": 3, "impaling": 5,
    "riptide": 3, "channeling": 1, "multishot": 1, "quick_charge": 3, "piercing": 4,
    "mending": 1, "vanishing_curse": 1
}