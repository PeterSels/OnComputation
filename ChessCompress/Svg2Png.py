import os
import glob
svg_list = glob.glob('*.svg')
print(svg_list)
for svg in svg_list:
    png = svg.replace('.svg', '.png')
    cmd = 'rsvg-convert -h 320 {:s} > {:s}'.format(svg, png)
    print(cmd)
    os.system(cmd)
