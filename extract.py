import zipfile
from PIL import Image
from io import BytesIO
import json, os, xxhash
from icon_3d import get_icon

zf = '/home/icicl/.minecraft/versions/1.16.1/1.16.1.jar'

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
        atlas = Image.new("RGBA", (16*(16*scale), 16*(16*scale)), (0,0,0,0))
        for y in range(16):
            for x in range(16):
                idx = 16*y+x
                if idx < len(all_items):
                    item = all_items[idx]
                    if is_item(item):
                        tex = get_texture(item, 'item').resize((16*scale, 16*scale), Image.Resampling.NEAREST).convert('RGBA')
                    else:
                        tex = get_icon(item, scale)
                    atlas.paste(tex, (x*16*scale, y*16*scale), tex)
        atlas.save(atlas_fp)    
    return atlas