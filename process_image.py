import os
from PIL import Image
from extract import get_chest_loot_table, get_texture, is_item
import zipfile
import re
from extract import load_all_tables, load_atlas
import numpy as np
from extract import load_ascii





def process_image(path, prompt_uncertain=True, verbose=False):
    def ocr(img_arr, x0, y0, alphabet, text_color, color_tolerance=1, max_consec_spaces=0):
        result = ""
        consec_spaces = -1
        while True:
            consec_spaces += 1
            match = None
            for char in alphabet:
                glyph = chars[char]
                dx = x0 - glyph.shape[1]
                if dx < 0: continue
                dy = y0
                masked = np.where(glyph, img_arr[dy:dy+glyph.shape[0],dx:dx+glyph.shape[1]].sum(axis=-1), 0).astype(int)
                if np.where(masked, abs(masked - (255 + 3*text_color)) <= 3*color_tolerance, 0).sum() == glyph.sum():
                    if match is None or glyph.sum() > match[1]:
                        match = (char, glyph.sum(), scale + + glyph.shape[1])
            if match:
                result += match[0]
                x0 -= match[2]
                consec_spaces = -1
            else:
                if consec_spaces == max_consec_spaces: break
                x0 -= 4*scale
                result += ' '
        return result[::-1].strip()


    def get_match(slot):
        slot_arr = np.array(slot)
        is_empty = abs(slot_arr - slot_arr[0,0]).sum() == 0
        if is_empty:
            return 1,None,0
        idx = 0
        best_score = 0.25
        best = None
        for x in range(0,16*16*scale, 16*scale):
            for y in range(0, 16*16*scale, 16*scale):
                if idx < len(all_items):
                    tex = atlas_arr[x:x+16*scale,y:y+16*scale]
                    mask = np.where(tex[:,:,-1] == 255, 1, 0)
                    diff_sq = (mask[:,:,None] * (tex - slot_arr))**2
                    near_matches = ((diff_sq.sum(axis=-1) <= 3) * mask).sum() # each pixel can miss by up to 1
                    score = near_matches / mask.sum()
                    if score > best_score:
                        best_score, best = score, all_items[idx]
                idx += 1
        qty = ocr(slot_arr, 16*scale, 16*scale-7*scale, '0123456789', 252)
        qty = int(qty) if qty else 1
        return round(best_score,3), best, qty


    if os.path.isdir(path):
        path = os.path.join(path, max(os.listdir(path)))
    if not path.lower().endswith('.png'):
        raise ValueError("File must end with .png")

    if verbose: print(f'Processing image {path}')
    im = Image.open(path).convert('RGBA')
    il = im.load()

    w,h = im.size
    scale = sum(il[x,h//2] == (0,0,0,255) for x in range(w//2))
    if not 1 <= scale <= 8:
        if verbose: print(f"Container GUI not detected in {path}.")
        return None
    if verbose: print(f"Detected GUI scale = {scale}.")

    for x1 in range(w): # Container GUI boundaries
        if il[x1, h//2] == (0,0,0,255): break
    for y1 in range(h):
        if il[w//2, y1] == (0,0,0,255): break


    all_tables = load_all_tables()
    all_items = set()
    for table in all_tables.values():
        for _,entries in table:
            for entry in entries:
                name = entry[0][0]
                if name is not None:
                    all_items.add(name)
    all_items = sorted(all_items)

    atlas = load_atlas(scale, all_items)
    atlas_arr = np.array(atlas)

    ascii_np = np.array(load_ascii().resize((8*16*scale, 8*16*scale), Image.Resampling.NEAREST))
    chars = {}
    for i in range(32, 128):
        y = i // 16
        x = i % 16
        chars[chr(i)] = np.where(ascii_np[8*y*scale:8*(y+1)*scale,8*scale*x:8*(x+1)*scale][:,:,-1] != 0, 1, 0)
    for c,carr in chars.items():
        chars[c] = carr[:np.argmax((carr.sum(axis=1) != 0) * np.arange(8*scale))+1,:np.argmax((carr.sum(axis=0) != 0) * np.arange(8*scale))+1]

    coords = ocr(np.array(im), w-3*scale, 92*scale, '0123456789,:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', 62, 13, 1)
    coords = re.findall(r'.*?(\d+), (\d+), (\d+)$', coords)
    if coords:
        ssx,ssy,ssz = map(int,coords[0])
        if verbose: print(f"Found targeted block coordinates: {ssx}, {ssy}, {ssz}.")
    else:
        if verbose: print("Failed to extract targeted block coords. Is F3 showing?")

    contents = []
    for cy in range(3):
        for cx in range(9):
            px = x1 + (1 + 2 + 4 + 1)*scale + (1 + 16 + 1)*scale*cx
            py = y1 + (1 + 2 +14 + 1)*scale + (1 + 16 + 1)*scale*cy
            slot = im.crop((px, py, px+16*scale, py+16*scale))
            conf,item,qty = get_match(slot)
            if item is None:
                if prompt_uncertain and conf != 1:
                    while True:
                        inp = input(f"Failed to detect item in row {cy}, column {cx}. Enter the item and quantity (ex. 'leather_chestplate,1'). Enter [S] to show the offending slot, or [F] to show the whole image.: ").replace(' ','')
                        if inp.count(',') == 1:
                            item,qty = inp.split(',')
                            if qty.isdecimal() and (item in all_items or 'minecraft:'+item in all_items): break
                        if inp.lower() == 's':
                            slot.show()
                        if inp.lower() == 'f':
                            im.show()
                    if item not in all_items: item = 'minecraft:' + item
                    contents.append((item,int(qty)))
                else:
                    contents.append((None,0))
            else: contents.append((item, qty))
    
    distinct_content_items = {}
    for item,qty in contents:
        if item is None: continue
        distinct_content_items[item] = qty + distinct_content_items.get(item, 0)
    if verbose: print(distinct_content_items)


    possible_tables = []
    for tname,table in all_tables.items():
        t_items = set()
        for _,entries in table:
            for entry in entries:
                if entry[0][0]: t_items.add(entry[0][0])
        if all(item in t_items for item in distinct_content_items):
            possible_tables.append(tname)
    if verbose: print(f"The detected loot can generate in the following structures: {possible_tables}.")

    im_cropped_to_container = im.crop((x1, y1, x1+scale*((1 + 2 + 4 + 1)*2 + (1 + 16 + 1)*9), y1+scale*((1 + 2 + 14 + 1)*2 + (1 + 16 + 1)*3)))
    return ((ssx,ssy,ssz) if coords else None), contents, distinct_content_items, possible_tables, im_cropped_to_container