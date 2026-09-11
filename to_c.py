import os, glob, ctypes, platform, sys
import numpy as np
from multiprocessing import Process

from process_image import process_image
from loot import loottable_c
from entropy_calc import entropy


salts = { # step, index, feat seed, grid-based spacing, enum index, spawn check type
    'desert_pyramid':(3,4,14357617,16*32,0,(0,32)),
    'buried_treasure':(1,3,10387320,16,1,(1,)),
#    'ruined_portal':(5,4,??,16*40,2,(0,40)),
#    'shipwreck_supply':(6,4),
#    'igloo_chest':(4,4),
#    'woodland_mansion':(1,4),
}

def print_table_row(vals,widths):
    if vals is None: # print the horizontal divider
        print('+ ' + '-+-'.join('-'*width for width in widths) + '-+')
    else:
        print('| ' + ' | '.join(f'{val:<{width}}' for val,width in zip(vals,widths)) + ' |')

def get_info(ss_dir, ss_count=10):
    candidates = []

    for file in sorted(glob.glob(ss_dir + '*.png'))[-ss_count:]:
        processed = process_image(file, prompt_uncertain=False)
        if processed is None: continue
        coords, contents, loot_item_counts, possible_tables, cropped = processed
        possible_tables = [table for table in possible_tables if sum(entropy(table,loot_item_counts,contents)) < float('inf')]
        candidates.append((file[len(ss_dir):],coords,possible_tables,loot_item_counts,contents))

    print(f"Detected {len(candidates)} screenshots with loot inventories:")
    col_widths = [4,30,50,18,18,30]
    print_table_row(None,col_widths)
    print_table_row(['Id#','Filename','Possible Structures','Block Coordinates','Chunk Coordinates','Entropy (I,S,F,R)'],col_widths)
    print_table_row(None,col_widths)
    for idx,(file,coords,possible_tables,loot,contents) in enumerate(candidates,1):
        print_table_row([
            idx,
            file,
            ', '.join(possible_tables),
            'Unknown' if coords is None else f'{coords[0]},{coords[2]}',
            'Unknown' if coords is None else f'{coords[0]//16},{coords[2]//16}',
            ', '.join(f'{h:.1f}' for h in entropy(possible_tables[0],loot,contents)) if len(possible_tables) == 1 else '????'
            ],col_widths)
    print_table_row(None,col_widths)
    print('Enter a comma-separated list of structures to use for filtering step.\n' \
        'The first structure will be used for the loot-based cracking.'
        'This step is much faster than the full loot-based cracker.\n' \
        'I recommend at least 16 bits of [F] entropy.\n' \
        'The structure with the largest [R] entropy will be used for fast iteration over reversal of initial PRNG calls.' \
        'You should have at least 48 bits of entropy [I]+[S] of the first entry + [R] of the entry with maximum [R] + the sum of [F] for all entries other than max [R].')
    spawn_checks = []
    while len(spawn_checks) == 0:
        while True:
            inp = input("Enter ID List: ").replace(' ','')
            if inp and all(i.isdecimal() for i in inp.split(',')):
                ids = list(map(int,inp.split(',')))
                if min(ids) >= 1 and max(ids) <= len(candidates):
                    break
        seen_cc = set()
        best_h4 = 0
        total_h3 = 0
        candidate_used_for_loot_cracking = None
        for idx in ids:
            file,coords,possible_tables,loot,contents = candidates[idx-1]
            table = possible_tables[0]
            if table not in salts:
                print(f"Skipping ID# {idx} - Currently only sturctures supported are: {', '.join(list(salts))}")
                continue
            if candidate_used_for_loot_cracking is None: candidate_used_for_loot_cracking = file
            if coords is None:
                print(f"Skipping ID# {idx} - Must have coordinates to use in filtering.")
                continue
            x,z = coords[0],coords[2]
            if len(possible_tables) != 1:
                print(f"Skipping ID# {idx} - Ambiguous structure type.")
                continue
            if (x//16,z//16) in seen_cc:
                print(f"Skipping ID# {idx} - Belongs to a structure already used for filtering.")
                continue
            _,_,h3,h4 = entropy(table, loot, contents)
            if h4 > best_h4:
                best_h4 = h4
                _,_,feat_seed_salt,grid_spacing,enum_idx,_ = salts[table]
                feat_seed_base = (((x // grid_spacing)*341873128712+(z // grid_spacing)*132897987541 + feat_seed_salt) & 0xFFFF_FFFF_FFFF) | (enum_idx << 48)
                skip_prng_reverse_feature = len(spawn_checks) if abs(h4 - h3) < 0.0001 else None
            spawn_check_type = salts[table][5][0]
            cx,cz = x//16,z//16
            if spawn_check_type == 0: # grid based w/ 2 randint calls
                spacing = salts[table][5][1]
                mcx,mcz = cx%spacing,cz%spacing
                rx,rz = cx//spacing,cz//spacing
                structure_salt = (rx*341873128712 + rz*132897987541 + salts[table][2]) & 0xFFFF_FFFF_FFFF
                structure_id = salts[table][4]
                spawn_checks.append(structure_salt | (mcz << 48) | (mcx << (48 + 6)) | (structure_id << (48 + 2*6)))
            elif spawn_check_type == 1: # every chunk, based on nextfloat call
                structure_salt = (cx*341873128712 + cz*132897987541 + salts[table][2]) & 0xFFFF_FFFF_FFFF
                structure_id = salts[table][4]
                spawn_checks.append(structure_salt | (structure_id << (48 + 2*6)))
            else:
                print(f"WARNING: unimplemented structure spawn check type {spawn_check_type}. Skipping...")
            seen_cc.add((x//16,z//16))
            total_h3 += h3
        if skip_prng_reverse_feature is not None: del spawn_checks[skip_prng_reverse_feature]
        if len(spawn_checks) == 0: print('At least one valid structure required for filter stage.')
    return candidate_used_for_loot_cracking, spawn_checks, feat_seed_base, (total_h3-best_h4, best_h4)


def run(ss_dir, ss_count):
    if not ss_dir.endswith('/'): ss_dir += '/'
    file, spawn_checks, feat_seed_base, (h3, h4) = get_info(ss_dir, ss_count)
    coords, contents, loot, possible_tables, cropped = process_image(ss_dir + file, prompt_uncertain=True)
    possible_tables = [table for table in possible_tables if sum(entropy(table,loot,contents)) < float('inf')]

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

    h1, h2, _, _ = entropy(table, loot, contents)
    step,index,feat_seed_salt,grid_spacing,enum_idx,_ = salts[table]
    shuffle_order = 0
    for i in range(27): shuffle_order |= ((contents[i][0] is not None) << i)
    loottable,lookup = loottable_c(table, loot)
    targ = [0]*len(lookup)
    for item,qty in loot.items(): targ[lookup[item]] = qty
    print(loot)

    print_c_code = False
    if print_c_code:
        print(f"Loot items give {h1:.1f} bits of information, and the positions of occupied chest slots gives {h2:.1f}. Have estimated {h3:.1f} spawn bits, and {h4:.1f} PRNG reversal bits.")
        print(f"Total: {h1+h2:.1f} bits of ~48 needed for unique determination.\n")
        print(f"uint32_t shuffle_order = 0b{bin(shuffle_order)[2:].zfill(27)};")
        print(f"uint8_t table[{len(loottable)}] = {'{'}{str(loottable)[1:-1]}{'}'};")
        print(f"uint8_t target[{len(targ)}] = {'{'}{str(targ)[1:-1]}{'}'};")
        print(f"int distinct_items = {len(lookup)};")
        print(f"int32_t feature_seed_info[4] = {'{'}{x}, {z}, {step}, {index}{'}'}; // x, z, index, step")
        print(f"int64_t prng_first_call_salt = {feat_seed_base}LL;")
        print(f'int64_t spawns_checks[{len(spawn_checks)}] = {'{'}{str(spawn_checks)[1:-1]}{'}'};')
        print(f"int num_spawn_checks = {len(spawn_checks)};")

    lib = ctypes.CDLL("./crack.so")
    lib.dispatch.restype = None
    lib.dispatch.argtypes = [
        ctypes.c_uint32,
        np.ctypeslib.ndpointer(dtype=np.uint8,  flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.uint8,  flags="C_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.int32,  flags="C_CONTIGUOUS"),
        ctypes.c_int64,
        np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS"),
        ctypes.c_int,
    ]

    def dispatch():
        lib.dispatch(
            shuffle_order,
            np.array(loottable, dtype=np.uint8),
            np.array(targ, dtype=np.uint8),
            len(lookup),
            np.array([x, z, step, index], dtype=np.int32),
            feat_seed_base,
            np.array(spawn_checks, dtype=np.int64),
            len(spawn_checks),
        )

    p = Process(target=dispatch)
    p.start()
    try:
        p.join()
    except KeyboardInterrupt:
        print("Issuing SIGKILL to dispatcher...")
        SIGKILL = 9
        os.kill(p.pid, SIGKILL)
        p.join()
    return True

#run('./ss/', 20)

#ss = '/home/icicl/.minecraft/screenshots/'
#ss = "./ss/"

def main():
    if len(sys.argv) == 1 or sys.argv[1] == "-":
        system = platform.system()
        if system == "Linux":
            path = os.path.join(os.getenv("HOME"), ".minecraft", "screenshots")
        elif system == "Windows":
            path = os.path.join(os.getenv("APPDATA"), ".minecraft", "screenshots")
        elif system == "Darwin":
            path = os.path.join(os.getenv("HOME"), "Library", "Application Support", "minecraft", "screenshots")
        else:
            path = None
    else:
        path = sys.argv[1]
    if not os.path.exists(path):
        print("Did not detect default minecraft folder.\nRun again with your screenshot directory as the first argument. Ex. 'python main.py ~/mc_screenshots/'")
        return
    run(path,20)

if __name__ == "__main__":
    main()
