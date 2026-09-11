from process_image import process_image
import os, glob

ss = '/home/icicl/.minecraft/screenshots/'
for file in sorted(glob.glob(ss + '*.png')):
    process_image(file)
    print()
