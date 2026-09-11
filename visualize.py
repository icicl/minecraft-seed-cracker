DIGITS_3x5 = {
    "0": [
        [1,1,1],
        [1,0,1],
        [1,0,1],
        [1,0,1],
        [1,1,1],
    ],
    "1": [
        [0,1,0],
        [1,1,0],
        [0,1,0],
        [0,1,0],
        [1,1,1],
    ],
    "2": [
        [1,1,1],
        [0,0,1],
        [1,1,1],
        [1,0,0],
        [1,1,1],
    ],
    "3": [
        [1,1,1],
        [0,0,1],
        [1,1,1],
        [0,0,1],
        [1,1,1],
    ],
    "4": [
        [1,0,1],
        [1,0,1],
        [1,1,1],
        [0,0,1],
        [0,0,1],
    ],
    "5": [
        [1,1,1],
        [1,0,0],
        [1,1,1],
        [0,0,1],
        [1,1,1],
    ],
    "6": [
        [1,1,1],
        [1,0,0],
        [1,1,1],
        [1,0,1],
        [1,1,1],
    ],
    "7": [
        [1,1,1],
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [0,0,1],
    ],
    "8": [
        [1,1,1],
        [1,0,1],
        [1,1,1],
        [1,0,1],
        [1,1,1],
    ],
    "9": [
        [1,1,1],
        [1,0,1],
        [1,1,1],
        [0,0,1],
        [1,1,1],
    ],
}

from PIL import Image
from extract import get_texture

def digit_to_image(digit, color=(0,0,0), pad=1, background = (255,255,255,255)):
    bitmap = DIGITS_3x5[digit]
    img = Image.new("RGBA", (3+2*pad, 5+2*pad), background)
    pixels = img.load()
    if len(color) == 3: color = color + (255,)
    for y in range(5):
        for x in range(3):
            pixels[x+pad, y+pad] = color if bitmap[y][x] else background
    return img


def visualize(loot, scale=4, w=9, h=3):
    assert w*h == len(loot)
    sz = 16
    im = Image.new("RGBA", (w*(sz+1)+1, h*(sz+1)+1), '#fff')
    il = im.load()
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            if y%(sz+1)==0 or x%(sz+1)==0:
                il[x,y] = (111,111,111)
    for y in range(h):
        for x in range(w):
            if loot[w*y+x]:
                item,qty = loot[w*y+x]
                tex = get_texture(item)
                im.paste(tex, (x*(sz+1)+1, y*(sz+1)+1), tex)
                qty_tex = digit_to_image(str(qty))
                qty_tex_w = digit_to_image(str(qty), (255,255,255))
                im.paste(qty_tex, (x*(sz+1)+12, y*(sz+1)+10), qty_tex)

    return im.resize((scale*im.size[0], scale*im.size[1]), 0)