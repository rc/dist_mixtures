#!/usr/bin/env python
import sys
import os
from glob import glob
import Image

def merge_images(filenames, n_row, n_col, sh):
    new_sh = (sh[0] * n_col, sh[1] * n_row)

    new_img = Image.new('RGB', new_sh, 'white')

    imgs = []
    for filename in filenames:
        print filename
        imgs.append(Image.open(filename))

    ii = 0
    for ir in range(n_row):
        for ic in range(n_col):
            new_img.paste(imgs[ii], (ic * sh[0], ir * sh[1]))
            ii += 1

    return new_img

def merge_selected():
    args = sys.argv[1:]

    n_row, n_col = int(args[0]), int(args[1])
    root_dir = args[2]

    print root_dir

    for dirname in os.listdir(os.path.abspath(root_dir)):
        dd = os.path.join(root_dir, dirname)
        if not os.path.isdir(dd): continue
        print dd

        filenames = glob(dd + '/*-data*') + sorted(glob(dd + '/*-fit*'))
        new_img = merge_images(filenames, n_row, n_col, (800, 600))

        merged_name = os.path.join(root_dir, dirname + '.png')
        print merged_name
        new_img.save(merged_name)

def merge_best():
    best = {
        '249_09' : [3, 3],
        '180_09' : [3, 3],
        '216_09' : [3, 3],
        '228_09' : [3, 3],
        '261_09' : [3, 3],
        '414_09' : [3, 3],
        '291_09' : [3, 3],
        '177_09' : [3, 3],
        '183_09' : [3, 3],
    }

    args = sys.argv[1:]
    root_dir = args[0]
    print root_dir

    for dirname in os.listdir(os.path.abspath(root_dir)):
        dd = os.path.join(root_dir, dirname)
        if not os.path.isdir(dd): continue
        print dd

        aux = glob(dd + '/*-data*')
        filenames = sorted(glob(dd + '/*-fit*'))
        dataset = os.path.basename(filenames[0]).split('-')[0]
        filenames = [ii for ii in filenames
                     for ic in best[dataset]
                     if int(ii.split('-')[-1][0]) == (ic - 1)]
        # Both are the same - no need to merge.
        new_img = merge_images(aux + filenames[:1], 1, 2, (800, 600))

        merged_name = os.path.join(root_dir, dirname + '-best.png')
        print merged_name
        new_img.save(merged_name)

def merge_ellipse_imagej():
    args = sys.argv[1:]

    root_dir = args[0]
    filenames = [sorted(glob(dd + '/*-data*') + glob(dd + '/*-fit-1*')
                        + glob(dd + '/*-fit-2*'))
                 for dd in args[1:]]

    npd = 3
    for ii in range(len(filenames[0]) / npd):
        names = (filenames[0][npd*ii:npd*ii+npd],
                 filenames[1][npd*ii:npd*ii+npd],
                 filenames[2][npd*ii:npd*ii+npd])
        nns = zip(*names)
        for ir in range(len(nns) / npd):
            nn = []
            for ik in range(npd):
                nn += nns[npd * ir + ik]
            new_img = merge_images(nn, npd, 3, (800, 600))

            dirname = os.path.basename(nn[0]).split('-')[0]
            merged_name = os.path.join(root_dir, dirname + '.png')
            print merged_name
            new_img.save(merged_name)

# merge_selected()
# merge_best()
# merge_ellipse_imagej()
