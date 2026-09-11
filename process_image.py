import os, zipfile, re, string
from PIL import Image
import numpy as np

from extract import load_all_tables, load_atlas, load_ascii


def get_chars():
    ascii_im = load_ascii().convert("RGBA")
    ascii_arr = np.array(ascii_im)
    chars = {}
    for charpoint in range(33,127):
        y = 8*(charpoint >> 4)
        x = 8*(charpoint & 0xF)
        glyph = ascii_arr[y:y+8,x:x+8]
        n = 0
        for i,v in enumerate((glyph[:,:,-1] == 255).astype(np.uint64).flatten()):
            n |= (v << i)
        for width in range(1,9):
            mask = int("01"*8,16) * ((1<<width) - 1)
            if n & mask == n: break
        chars[chr(charpoint)] = (n,width,mask)
    return chars


def ocr(im, stride=None, xslice=slice(None), yslice=slice(None)):
    if type(im) is str: im = Image.open(path)
    im = im.convert("RGB")

    texts = []
    px0 = np.array(im)
    px0 = np.where((px0.max(axis=-1)==px0.min(axis=-1)), px0[:,:,0], 0)
    px0 = np.where((px0 >= 56), 1, 0).astype(np.uint8)
    Image.fromarray(px0*255).save('tmp.png')
    for stride in (range(4,0,-1) if stride is None else (stride,)):
        px = px0[::stride,::stride]
        px = px[yslice,xslice]

        psum_8 = np.zeros_like(px, dtype=np.uint64)
        psum_8 += px
        for i in range(1,8):
            psum_8[:,:-i] += px[:,i:] << i
        psum_64 = np.zeros_like(px, dtype=np.uint64)
        psum_64 += psum_8
        for i in range(1,8):
            psum_64[:-i,:] += psum_8[i:,:] << np.uint64(8*i)

        hits = {}
        masked_cache = {} # avoid recomputing masked array every time
        for c,(n,w,m) in chars.items():
            m = m | (m << 1) # whitespace on right
            if w < 8: m,n = m | (m << 1), n << 1 # whitespace on left
            if m not in masked_cache: masked_cache[m] = (psum_64 & m)
            hy,hx = np.where(masked_cache[m] == n)
            for y,x in zip(hy,hx): hits[y,x] = c

        lines = {}
        for (y,x),c in sorted(hits.items()):
            if y not in lines: lines[y] = []
            lines[y].append((x,c))
        text = ''
        for y,line in lines.items():
            line.sort()
            s = ''
            targ_x = 0
            for x,c in line:
                if x != targ_x:
                    s += ' '
                    targ_x = x
                s += c
                targ_x += chars[c][1] + 1
            if set(string.ascii_letters + string.digits) & set(s): text += s.strip() + '\n' # ignore unaccompanied ,.|;: type hits 
        texts.append(text)
    return max(texts,key=len)

chars = get_chars()


def get_coords(im, scale=None):
    text = ocr(im, stride=scale, xslice=slice(-300,None), yslice=slice(80,100))
    coords = re.findall(r'\-?\d+, \-?\d+, \-?\d+', text)
    if coords: return list(map(int,coords[0].split(', ')))
    text = ocr(im, 2)
    print(text)
    coords = re.findall(r'\-?\d+, \-?\d+, \-?\d+', text)
    if coords: return list(map(int,coords[0].split(', ')))
    return None


def process_image(path, prompt_uncertain=True, verbose=False):
    def ocr_qty(img_arr, x0, y0, alphabet, text_color, color_tolerance=1, max_consec_spaces=0):
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
        slot_mask = np.where(abs(slot_arr - slot_arr[0,0]).sum(axis=-1) != 0, True, False)
        is_empty = abs(slot_arr - slot_arr[0,0]).sum() == 0
        if is_empty:
            return 1,None,0
        idx = 0
        best_score = 0.25
        best = None
        best_silh = (0,None)
        for x in range(0,16*16*scale, 16*scale):
            for y in range(0, 16*16*scale, 16*scale):
                if idx < len(all_items):
                    tex = atlas_arr[x:x+16*scale,y:y+16*scale]
                    mask = np.where(tex[:,:,-1] == 255, True, False)
                    diff_sq = (mask[:,:,None] * (tex - slot_arr))**2
                    near_matches = ((diff_sq.sum(axis=-1) <= 3) * mask).sum() # each pixel can miss by up to 1
                    score = near_matches / mask.sum()
                    mask_similarity = (slot_mask*mask).sum() / (mask | slot_mask).sum()
                    if score > best_score:
                        best_score, best = score, all_items[idx]
                    best_silh = min(best_silh, (-mask_similarity, all_items[idx]))
                idx += 1
        qty = ocr_qty(slot_arr, 16*scale, 16*scale-7*scale, '0123456789', 252)
        qty = int(qty) if qty else 1
        if best is None and best_silh[0] == -1:
            return -1, best_silh[1], qty
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


#    coords = get_coords(im, scale)
#    if coords:
#        ssx,ssy,ssz = coords
    coords = ocr_qty(np.array(im), w-3*scale, 92*scale, '-0123456789,', 62, 13, 1)
    coords = re.findall(r'.*?(\-?\d+), (\-?\d+), (\-?\d+)$', coords)
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
            if conf == -1:
                print(f"Detected enchanted item {item} in row {cy+1}, column {cx+1}. Please ensure this is correct, as the enchantment glint can interfere with detection.")
            if item is None:
                if prompt_uncertain and conf != 1:
                    while True:
                        inp = input(f"Failed to detect item in row {cy+1}, column {cx+1}. Enter the item and quantity (ex. 'leather_chestplate,1'). Enter [S] to show the offending slot, or [F] to show the whole image.: ").replace(' ','')
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