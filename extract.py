import zipfile
from PIL import Image
from io import BytesIO
import json, os, xxhash, time
from loottable import load_table

os.makedirs('cache/', exist_ok=True)

zf = '/home/icicl/.minecraft/versions/1.16.1/1.16.1.jar'
zf = './cache/1.16.1.jar'
if not os.path.exists(zf):
    import requests
    url = 'https://piston-data.mojang.com/v1/objects/c9abbe8ee4fa490751ca70635340b7cf00db83ff/client.jar'
    print(f"Downloading Minecraft .jar...")
    data = requests.get(url).content
    with open(zf, 'wb') as f: f.write(data)

table_cache = {}
def get_chest_loot_table(chest, save=False):
    if chest in table_cache: return table_cache[chest]
    path = f'data/minecraft/loot_tables/chests/{chest}.json'
    with zipfile.ZipFile(zf, 'r') as z:
        with z.open(path) as f:
            content = f.read().decode("utf-8")
            if save:
                with open(path.split('/')[-1], 'w') as f: f.write(content)
            result = json.loads(content)
            table_cache[chest] = result
            return result


def is_item(name):
    name = name.replace('minecraft:','')
    with zipfile.ZipFile(zf) as z:
        return f'assets/minecraft/textures/item/{name}.png' in z.namelist()

texture_cache = {}

def get_texture(item, blockface=None):
    item = item.replace('minecraft:','')
    if (item,blockface) in texture_cache: return texture_cache[item,blockface]
    if blockface is None:
        paths = [
            f'assets/minecraft/textures/item/{item}.png',
            f'assets/minecraft/textures/block/{item}.png',
            f'assets/minecraft/textures/block/{item}_side.png',
            f'assets/minecraft/textures/block/bedrock.png',
        ]
    elif blockface == 'top':
        paths = [
            f'assets/minecraft/textures/block/{item}_top.png',
            f'assets/minecraft/textures/block/{item}.png',
            f'assets/minecraft/textures/block/bedrock.png',
        ]
    elif blockface == 'side':
        paths = [
            f'assets/minecraft/textures/block/{item}_side.png',
            f'assets/minecraft/textures/block/{item}.png',
            f'assets/minecraft/textures/block/bedrock.png',
        ]
    elif blockface == 'item':
        paths = [
            f'assets/minecraft/textures/item/{item}.png',
            f'assets/minecraft/textures/block/bedrock.png',
        ]
    else:
        raise ValueError
    for path in paths:
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                with z.open(path) as f:
                    content = f.read()
            im = Image.open(BytesIO(content))
            texture_cache[item,blockface] = im
            return im
        except:
            pass
    raise ValueError


def load_all_tables():
    all_table_fp = 'cache/all_tables.txt'
    if os.path.exists(all_table_fp):
        with open(all_table_fp) as f: tables = eval(f.read())
    else:
        print("Cache for processed loot tables not found on disk. Regenerating...")
        tables = {}
        with zipfile.ZipFile(zf, 'r') as z:
            for zi in z.filelist:
                if (file := zi.filename).startswith('data/minecraft/loot_tables/chests/'):
                    chest = file[len('data/minecraft/loot_tables/chests/'):-5]
                    if chest.startswith('village/'): continue
                    table = (get_chest_loot_table(chest))
                    table = load_table(table)
                    tables[chest] = table
        with open(all_table_fp, 'w') as f: f.write(str(tables))
    return tables

def load_atlas(scale, itemlist):
    assert len(itemlist) <= 256
    itemlist = sorted(itemlist)
    hash = xxhash.xxh32_hexdigest(str(itemlist))
    atlas_fp = f'cache/atlas_scale{scale}_{hash}.png'
    if os.path.exists(atlas_fp):
        atlas = Image.open(atlas_fp)
    else:
        t0 = time.time()
        t1 = t0
        print(f"Atlas for GUI scale {scale} not found on disk. Regenerating, please wait ~1-2min...")
        atlas = Image.new("RGBA", (16*(16*scale), 16*(16*scale)), (0,0,0,0))
        for y in range(16):
            for x in range(16):
                idx = 16*y+x
                if idx < len(itemlist):
                    item = itemlist[idx]
                    if is_item(item):
                        tex = get_texture(item, 'item').convert('RGBA')
                        texl = np.array(tex)
                        texl[:,:,:3] = (texl[:,:,:3]*0.9875).round()
                        tex = Image.fromarray(texl)
                        tex = tex.resize((16*scale, 16*scale), Image.Resampling.NEAREST)
                    else:
                        tex = get_icon(item, scale)
                    atlas.paste(tex, (x*16*scale, y*16*scale), tex)
                if (time.time() - t1 > 10):
                    t1 = time.time()
                    print(f"    [{t1 - t0:5.1f}s] Processed {idx} of {len(itemlist)} atlas textures.")
        atlas.save(atlas_fp)    
    return atlas

def load_ascii():
    path = 'assets/minecraft/textures/font/ascii.png'
    with zipfile.ZipFile(zf, 'r') as z:
        with z.open(path) as f:
            content = f.read()
    return Image.open(BytesIO(content))



####### ICON3D #######
import numpy as np
from PIL import Image

vertices = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],  # front
    [0,0,1],[1,0,1],[1,1,1],[0,1,1]   # back
], dtype=float)

uv_idx = np.array([
    [0,0],[0,1],[1,1],[1,0]
])

vertices -= 0.5
ROT_X = np.radians(30)
ROT_Y = np.radians(225)

Rx = np.array([
    [1,0,0],
    [0,np.cos(ROT_X),-np.sin(ROT_X)],
    [0,np.sin(ROT_X),np.cos(ROT_X)]
])

Ry = np.array([
    [np.cos(ROT_Y),0,np.sin(ROT_Y)],
    [0,1,0],
    [-np.sin(ROT_Y),0,np.cos(ROT_Y)]
])

vertices = vertices @ Ry.T
vertices = vertices @ Rx.T
vertices *= 0.625

faces = {
    "top": [3,2,6,7],
    "left": [0,3,2,1],
    "right": [1,2,6,5],
}

from functools import cache
@cache
def getuv(x,y):
    P = np.array([x,y]) - 0.5
    for fname,fidx in faces.items():
        for face,uvface in zip((fidx[0:3], fidx[2:]+fidx[:1]),([0,1,2],[2,3,0])):
            A,B,C = vertices[face][:,:2]
            v0,v1,v2 = B-A,C-A,P-A
            def cross(p1,p2): return p1[0]*p2[1]-p1[1]*p2[0]
            den = cross(v0,v1)
            w1 = cross(v2, v1) / den
            w2 = cross(v0, v2) / den
            w0 = 1 - w1 - w2
            w = np.array([w0,w1,w2])
            if min(w) >= 0:
                return (fname,w@uv_idx[uvface])
    return None

def get_icon(block, scale):
    tex_side = get_texture(block, 'side').transpose(Image.FLIP_TOP_BOTTOM).load()
    tex_top = get_texture(block, 'top').transpose(Image.FLIP_TOP_BOTTOM).load()

#    im = Image.new("RGBA", (16*scale, 16*scale), (0,0,0,0))
#    il = im.load()

    pixels = np.zeros((16*scale, 16*scale, 4), dtype=np.uint8)

    for x in range(16*scale):
        for y in range(16*scale):
            uv = getuv((x+0.5)/(16*scale),(y+0.5)/(16*scale))
            if uv:
                f,(u,v) = uv
                tex = tex_top if f == 'top' else tex_side
                dim = {'top':0.91, 'left':0.69, 'right':0.40}[f] # estimates
                pxl = tex[round(u*16-0.5), round(v*16-0,5)]
                pixels[x,y] = (round(pxl[0]*dim), round(pxl[1]*dim), round(pxl[2]*dim), 255)
                
    return Image.fromarray(pixels).transpose(Image.ROTATE_180).convert("RGBA")
