import zipfile, json, os, xxhash, time
from PIL import Image
from io import BytesIO

from loottable import load_table
from functools import cache
from icon3d import get_icon

os.makedirs('cache/', exist_ok=True)
zf = './cache/1.16.1.jar' # TODO - specify which version as argument
if not os.path.exists(zf):
    import requests
    url = 'https://piston-data.mojang.com/v1/objects/c9abbe8ee4fa490751ca70635340b7cf00db83ff/client.jar'
    print(f"Downloading Minecraft .jar...")
    data = requests.get(url).content
    with open(zf, 'wb') as f: f.write(data)


@cache
def get_chest_loot_table(chest):
    path = f'data/minecraft/loot_tables/chests/{chest}.json'
    with zipfile.ZipFile(zf, 'r') as z:
        with z.open(path) as f:
            content = f.read().decode("utf-8")
            result = json.loads(content)
            return result


@cache
def get_texture(item, blockface=None):
    item = item.replace('minecraft:','')
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
            return Image.open(BytesIO(content))
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


def is_item(name):
    name = name.replace('minecraft:','')
    with zipfile.ZipFile(zf) as z:
        return f'assets/minecraft/textures/item/{name}.png' in z.namelist()


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
