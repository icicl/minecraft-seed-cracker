import os, glob, ctypes, platform, sys, time, shutil, re
import numpy as np
from multiprocessing import Process
from datetime import datetime

from process_image import process_image
from loot import loottable_c
from entropy_calc import entropy
from enchants import get_enchant_rng_call_data_c


salts = { # step, index, feat seed, grid-based spacing, enum index, spawn check type
    'desert_pyramid':(3,4,14357617,16*32,0,(0,32)),
    'buried_treasure':(1,3,10387320,16,1,(1,)),
    'ruined_portal':(5,4,34222645,16*40,2,(0,40)),
#    'shipwreck_supply':(6,4),
#    'igloo_chest':(4,4),
#    'woodland_mansion':(1,4),
#    'jungle_temple':(,,14357619,16*32,,(0,32))
}

def print_container(contents):
    cols = shutil.get_terminal_size().columns
    cell_w = (cols - 10) // 9
    max_w = 10
    cell_w = min(max_w, cell_w)
    print('+' + '+'.join('-'*cell_w for _ in range(9)) + '+')
    for row in range(3):
        words = [[] for _ in range(9)]
        for idx,(item,qty) in enumerate(contents[9*row:9*(row+1)]):
            if item is None: continue
            for word in item.replace('minecraft:','').split('_'):
                words[idx].append(word)
            words[idx].append(str(qty))
        for i in range(cell_w//2):
            print('|' + '|'.join(f'{words[j][i][:cell_w]: ^{cell_w}}' if len(words[j])>i else ' '*cell_w for j in range(9)) + '|')
        print('+' + '+'.join('-'*cell_w for _ in range(9)) + '+')
        
            


def print_table_row(vals,widths):
    if vals is None: # print the horizontal divider
        print('+ ' + '-+-'.join('-'*width for width in widths) + '-+')
    else:
        print('| ' + ' | '.join(f'{val:<{width}}' for val,width in zip(vals,widths)) + ' |')

def get_info(ss_dir, ss_count=10):
    t0 = time.time()
    candidates = []

    col_widths = [4,30,50,18,18,30]
    print_table_row(None,col_widths)
    print_table_row(['Id#','Filename','Possible Structures','Block Coordinates','Chunk Coordinates','Entropy (F,R)'],col_widths)
    print_table_row(None,col_widths)
    for file in sorted(glob.glob(ss_dir + '*.png'))[-ss_count:]:
        processed = process_image(file, prompt_uncertain=False)
        if processed is None: continue
        coords, contents, possible_tables = processed
        possible_tables = [table for table in possible_tables if sum(entropy(table)) < float('inf')]
        candidates.append((file[len(ss_dir):],coords,possible_tables))
        print_table_row([
            len(candidates),
            file[len(ss_dir):],
            ', '.join(possible_tables),
            'Unknown' if coords is None else f'{coords[0]},{coords[2]}',
            'Unknown' if coords is None else f'{coords[0]//16},{coords[2]//16}',
            ', '.join(f'{h:4.1f}' for h in entropy(possible_tables[0])) if len(possible_tables) == 1 else '????'
            ],col_widths)
    print_table_row(None,col_widths)

    print_table_row([f"Detected {len(candidates)} screenshots with loot inventories:"], [sum(col_widths)+3*len(col_widths)-3])
    print_table_row(None,col_widths)
    print(f"{time.time() - t0:.2f}s")
    print('\nEnter a comma-separated list of structures to use for filtering step.\n' \
        'The first structure will be used for the loot-based cracking.\n'
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
        candidate_used_for_loot_cracking = None
        for idx in ids:
            file,coords,possible_tables = candidates[idx-1]
            if len(possible_tables) != 1:
                print(f"Skipping ID# {idx} - Ambiguous structure type.")
                continue
            table = possible_tables[0]
            if table not in salts:
                print(f"Skipping ID# {idx} - Currently only sturctures supported are: {', '.join(list(salts))}")
                continue
            if candidate_used_for_loot_cracking is None: candidate_used_for_loot_cracking = file
            if coords is None:
                print(f"Skipping ID# {idx} - Must have coordinates to use in filtering.")
                continue
            x,z = coords[0],coords[2]
            spawn_checks.append([x, z, table])
        if len(spawn_checks) == 0: print('At least one valid structure required for filter stage.')
    return candidate_used_for_loot_cracking, spawn_checks


def process_info_for_cuda(spawn_checks_in, loot_container, loot_spawn):
    seen_cc = set()
    best_h4 = 0
    total_h3 = 0
    spawn_checks = []
    for x,z,table in spawn_checks_in:
        cx,cz = x//16,z//16
        if (cx,cz) in seen_cc:
            print(f"Skipping {table} as {cx},{cz} - Belongs to a structure already used for filtering.")
            continue
        h3,h4 = entropy(table)
        if h4 > best_h4:
            best_h4 = h4
            _,_,feat_seed_salt,grid_spacing,enum_idx,_ = salts[table]
            feat_seed_base = (((x // grid_spacing)*341873128712+(z // grid_spacing)*132897987541 + feat_seed_salt) & 0xFFFF_FFFF_FFFF) | (enum_idx << 56)
            spawn_check_type = salts[table][5][0]
            if spawn_check_type == 0:
                spacing = salts[table][5][1]
                feat_seed_base |= ((x // 16) % spacing) << 48
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
    if skip_prng_reverse_feature is not None: del spawn_checks[skip_prng_reverse_feature]

    table,x,z = loot_spawn

    loot = {}
    for item,qty in loot_container:
        if item is None: continue
        loot[item] = qty + loot.get(item, 0)

    step,index,feat_seed_salt,_,enum_idx,_ = salts[table]
    shuffle_order = 0
    for i in range(27): shuffle_order |= ((loot_container[i][0] is not None) << i)
    loottable,lookup = loottable_c(table, loot)
    targ = [0]*len(lookup)
    for item,qty in loot.items(): targ[lookup[item]] = qty
    split_stacks = [((lookup[item] << 8) | qty) if item else 0 for item,qty in loot_container]
    ench_callcounts = get_enchant_rng_call_data_c()
    print(f"\nCracking structure {table} loot chest with these contents:")
    print_container(loot_container)
    for x,z,table in spawn_checks_in:
        print(f"Filtering seeds using structure spawn of type {table} as coordinates {x},{z}")
    print()

    return (shuffle_order,
            np.array(loottable, dtype=np.uint8),
            np.array(targ, dtype=np.uint8),
            len(lookup),
            np.array([x, z, step, index], dtype=np.int32),
            feat_seed_base,
            np.array(spawn_checks, dtype=np.int64),
            len(spawn_checks),
            np.array(split_stacks, dtype=np.uint16),
            np.array(ench_callcounts, dtype=np.uint64))

def run(ss_dir, ss_count):
    if os.path.isdir(ss_dir):
        if not ss_dir.endswith('/'): ss_dir += '/'
        file, spawn_checks = get_info(ss_dir, ss_count)
        coords, contents, possible_tables = process_image(ss_dir + file, prompt_uncertain=True)
    #    possible_tables = [table for table in possible_tables if sum(entropy(table)) < float('inf')] TODO replace (fast H calc)

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

        os.makedirs('run_history', exist_ok=True)
        with open(f'run_history/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.txt', 'w') as f:
            for scx,scz,sctable in spawn_checks:
                f.write(f'{scx}\t{scz}\t{sctable}\n')
            f.write(f'\n{x}\t{z}\t{table}\n')
            for row in range(3):
                for item,qty in contents[row*9:row*9+9]:
                    if item is None:
                        f.write('-\n')
                    else:
                        f.write(f'{item.replace("minecraft:","")}\t{qty}\n')
                f.write('\n')
    else:
        spawn_checks = []
        contents = []
        with open(ss_dir) as f:
            for line_ in f:
                if line_ == '\n': break
                line = line_.strip().split()
                if len(line) == 3 and re.fullmatch(r'\-?\d+', line[0]) and re.fullmatch(r'\-?\d+', line[1]) and line[2].lower() in salts:
                    spawn_checks.append([int(line[0]), int(line[1]), line[2].lower()])
                else:
                    print(f"Skipping line '{line_.strip()}', does not match format 'x z structure'.")
            line_ = f.readline().strip()
            line = line_.split()
            if len(line) == 3 and re.fullmatch(r'\-?\d+', line[0]) and re.fullmatch(r'\-?\d+', line[1]) and line[2].lower() in salts:
                x,z,table = int(line[0]), int(line[1]), line[2].lower()
            else:
                print(f"Skipping line '{line_}' - does not match format 'x z structure'.")
            while len(contents) < 27:
                line_ = f.readline().strip()
                if line_ == "": continue
                line = line_.split()
                if len(line) == 2 and re.fullmatch(r'\d+', line[1]):
                    contents.append(['minecraft:' + line[0].lower(), int(line[1])]) # TODO check item valid
                else:
                    contents.append([None, 0])
                


    cuda_data = process_info_for_cuda(spawn_checks, contents, [table, x, z])

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
        np.ctypeslib.ndpointer(dtype=np.uint16, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS"),
    ]

    def dispatch():
        lib.dispatch(
            *cuda_data
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
