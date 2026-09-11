

# The order here is critical to matching the Registry next_int results
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

ENCHANT_DATA = {
    # id: (max_level, category)
    "protection": (4, "ARMOR"), "fire_protection": (4, "ARMOR"), 
    "feather_falling": (4, "ARMOR_FEET"), "blast_protection": (4, "ARMOR"),
    "projectile_protection": (4, "ARMOR"), "respiration": (3, "ARMOR_HEAD"),
    "aqua_affinity": (1, "ARMOR_HEAD"), "thorns": (3, "ARMOR"),
    "depth_strider": (3, "ARMOR_FEET"), "frost_walker": (2, "ARMOR_FEET"),
    "binding_curse": (1, "WEARABLE"), "sharpness": (5, "WEAPON"),
    "smite": (5, "WEAPON"), "bane_of_arthropods": (5, "WEAPON"),
    "knockback": (2, "WEAPON"), "fire_aspect": (2, "WEAPON"),
    "looting": (3, "WEAPON"), "sweeping": (3, "WEAPON"),
    "efficiency": (5, "DIGGER"), "silk_touch": (1, "DIGGER"),
    "unbreaking": (3, "BREAKABLE"), "fortune": (3, "DIGGER"),
    "power": (5, "BOW"), "punch": (2, "BOW"), "flame": (1, "BOW"),
    "infinity": (1, "BOW"), "luck_of_the_sea": (3, "FISHING_ROD"),
    "lure": (3, "FISHING_ROD"), "loyalty": (3, "TRIDENT"),
    "impaling": (5, "TRIDENT"), "riptide": (3, "TRIDENT"),
    "channeling": (1, "TRIDENT"), "multishot": (1, "CROSSBOW"),
    "quick_charge": (3, "CROSSBOW"), "piercing": (4, "CROSSBOW"),
    "mending": (1, "BREAKABLE"), "vanishing_curse": (1, "BREAKABLE")
}

MASTER_REGISTRY = [
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

def can_enchant(enchant, item_id):
    """
    Perfectly mimics Minecraft's Enchantment.canEnchant(ItemStack) logic,
    including specific class overrides.
    """
    if "enchanted_book" in item_id:
        return True
        
    # BREAKABLE (Everything with durability gets these)
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

    # WEAPONS & DIGGERS (The Axe Exception)
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

def get_random_enchant(item_id, rng):
    enchantments = [e for e in MASTER_REGISTRY if can_enchant(e, item_id)]
#    print(enchantments)
    idx = rng.next_int(len(enchantments))
    enchant = enchantments[idx]
    return enchant,ENCHANT_LEVELS[enchant]


