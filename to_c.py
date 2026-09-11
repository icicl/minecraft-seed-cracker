from process_image import process_image
from visualize import visualize
from extract import load_all_tables
import os, glob

salts = { # step, index, feat seed, grid-based spacing, enum index
    'buried_treasure':(1,3,10387320,16,1),
    'desert_pyramid':(3,4,14357617,16*32,0),
    'ruined_portal':(5,4),
#    'shipwreck_supply':(6,4),
#    'igloo_chest':(4,4),
#    'woodland_mansion':(1,4),
}

ss = '/home/icicl/.minecraft/screenshots/'
for file in sorted(glob.glob(ss + '*.png'))[1:2]:
    coords, contents, loot_item_counts, cropped = process_image(file)
    print()

#from widget import launch
#launch(visualize(contents), cropped)

possible_tables = []
for tname,table in load_all_tables().items():
    t_items = set()
    for _,entries in table:
        for entry in entries:
            if entry[0][0]: t_items.add(entry[0][0])
    if all(item in t_items for item in loot_item_counts):
        possible_tables.append(tname)
print(f"The detected loot can generate in the following structures: {possible_tables}.")



if coords is None:
    while True:
        xz = input("Failed to extract coordinates from screenshot (was F3 shown?). Please enter x,z of targeted block (chest position).")
        if xz.count(',') == 1:
            x,z = xz.split(',')
            if x.isdecimal() and z.isdecimal:
                x,z = int(x),int(z)
                break
else:
    x,y,z = coords


assert len(possible_tables) >= 1
if len(possible_tables) > 1:
    while True:
        tidx = input("Please select the structure type:\n" + '\n'.join(f'  [{i+1}] {t}' for i,t in enumerate(possible_tables)) + '\nEnter number:')
        if tidx.isdecimal() and 1 <= int(tidx) <= len(possible_tables):
            table = possible_tables[int(tidx)-1]
            break
else:
    table = possible_tables[0]

from loot import loottable_c
from entropy_calc import entropy

h1, h2 = entropy(table, loot_item_counts, contents)
print(f"Loot items give {h1:.1f} bits of information, and the positions of occupied chest slots gives {h2:.1f}.")
print(f"Total: {h1+h2:.1f} bits of ~48 needed for unique determination.\n")
step,index,feat_seed_salt,grid_spacing,enum_idx = salts[table]
feat_seed_base = ((x // grid_spacing)*341873128712+(z // grid_spacing)*132897987541 + feat_seed_salt) & 0xFFFF_FFFF_FFFF
shuffle_order = 0
for i in range(27): shuffle_order |= ((contents[i][0] is not None) << i)
print(f"uint32_t shuffle_order = 0b{bin(shuffle_order)[2:].zfill(27)};")
loottable,lookup = loottable_c(table, loot_item_counts)
print(f"uint8_t table[{len(loottable)}] = {'{'}{str(loottable)[1:-1]}{'}'};")
targ = [0]*len(lookup)
for item,qty in loot_item_counts.items(): targ[lookup[item]] = qty
print(f"uint8_t target[{len(targ)}] = {'{'}{str(targ)[1:-1]}{'}'};")
print(f"int distinct_items = {len(lookup)};")
print(f"int32_t feature_seed_info[4] = {'{'}{x}, {z}, {step}, {index}{'}'}; // x, z, index, step")
print(f"int64_t buried_treasure_float = {(enum_idx << 48)|feat_seed_base}LL;")