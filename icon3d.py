import numpy as np
from PIL import Image
from functools import cache

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
