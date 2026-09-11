import zipfile
from PIL import Image
from io import BytesIO
import json

zf = '/home/icicl/.minecraft/versions/1.16.1/1.16.1.jar'
def get_chest_loot_table(chest, save=False):
    path = f'data/minecraft/loot_tables/chests/{chest}.json'
    with zipfile.ZipFile(zf, 'r') as z:
        with z.open(path) as f:
            content = f.read().decode("utf-8")
            if save:
                with open(path.split('/')[-1], 'w') as f: f.write(content)
            return json.loads(content)


texture_cache = {}

def get_texture(item):
    item = item.replace('minecraft:','')
    if item in texture_cache: return texture_cache[item]
    for path in [
        f'assets/minecraft/textures/item/{item}.png',
        f'assets/minecraft/textures/block/{item}.png',
        f'assets/minecraft/textures/block/{item}_side.png',
        f'assets/minecraft/textures/block/bedrock.png',
    ]:
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                with z.open(path) as f:
                    content = f.read()
            im = Image.open(BytesIO(content))
            texture_cache[item] = im
            return im
        except:
            pass
    raise ValueError