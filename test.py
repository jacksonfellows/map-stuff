import matplotlib
from matplotlib import pyplot as plt

import rioxarray as rxr

data = rxr.open_rasterio('sample.tif', masked=True)
# data.plot()
# plt.show()

import skimage
import skimage.measure

def find_flats(d):
    return skimage.measure.label(d[0], connectivity=1)

import random

def random_cmap():
    colors = [(1,1,1)] + [(random.random(),random.random(),random.random()) for i in range(255)]
    return matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)

labels = find_flats(data)
# plt.imshow(labels, cmap=random_cmap())
# plt.show()

def draw_big_regions(d, l):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(d[0])

    for region in skimage.measure.regionprops(l):
        # take regions with large enough areas
        if region.area >= 30:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    # ax.set_axis_off()
    # plt.tight_layout()
    plt.show()

# draw_big_regions(data, labels)

def region_info(l):
    return list(skimage.measure.regionprops(labels, intensity_image=data[0].values))

regions = region_info(labels)

import numpy as np

def pad_edges(d):
    padded = np.pad(data[0], 1, constant_values=-1)
    return np.ma.masked_array(padded, mask=padded==-1)

padded = pad_edges(data)

# plt.imshow(padded)
# plt.show()

# padded

import scipy

CROSS = np.array(((0,1,0),(1,1,1),(0,1,0)), dtype=bool)

def find_features(d, p, rs):
    features = []
    for region in rs:
        image = np.pad(region.image.astype(np.uint8), 1)
        dilated = scipy.ndimage.binary_dilation(image, CROSS, border_value=0)
        bounds = dilated - image
        new_slices = tuple(slice(s.start,s.stop+2,s.step) for s in region.slice)
        data_im = p[new_slices]
        edge = np.ma.masked_array(data_im, np.logical_not(bounds), hard_mask=True)
        minr, minc, maxr, maxc = region.bbox
        if np.count_nonzero(bounds) != np.count_nonzero(edge.filled(0)):
            features.append(('edge', region.centroid))
        elif np.all(edge < region.min_intensity):
            features.append(('peak', region.centroid))
        elif np.all(edge > region.max_intensity):
            features.append(('pit', region.centroid))
        else:
            new_edge = edge.copy()
            new_edge[edge > region.max_intensity] = 2
            new_edge[edge < region.min_intensity] = 1
            new_edge = new_edge.filled(-1)
            groups, num = skimage.measure.label(new_edge, connectivity=2, return_num=True, background=-1)
            assert num > 1
            assert num % 2 == 0
            if num > 2:
                features.append(('saddle', region.centroid))
    return features

def find_features_test(d, p, rs):
    features = []
    for i,region in enumerate(rs):
        image = np.pad(region.image.astype(np.uint8), 1)
        dilated = scipy.ndimage.binary_dilation(image, CROSS, border_value=0)
        bounds = dilated - image
        new_slices = tuple(slice(s.start,s.stop+2,s.step) for s in region.slice)
        data_im = p[new_slices]
        edge = np.ma.masked_array(data_im, np.logical_not(bounds), hard_mask=True)
        minr, minc, maxr, maxc = region.bbox
        if np.count_nonzero(bounds) != np.count_nonzero(edge.filled(0)):
            features.append(('edge', region.centroid))
        elif np.all(edge < region.min_intensity):
            print('peak', i)
            features.append(('peak', region.centroid))
        elif np.all(edge > region.max_intensity):
            features.append(('pit', region.centroid))
        else:
            new_edge = edge.copy()
            new_edge[edge > region.max_intensity] = 2
            new_edge[edge < region.min_intensity] = 1
            new_edge = new_edge.filled(-1)
            groups, num = skimage.measure.label(new_edge, connectivity=2, return_num=True, background=-1)
            assert num > 1
            assert num % 2 == 0
            if num > 2:
                print('saddle', i)
                features.append(('saddle', region.centroid))
    return features

CELL = np.array(((0,1,0),(1,0,1),(0,1,0)), dtype=bool)

SADDLE_A = np.array((1,0,1,0), dtype=bool)
SADDLE_B = np.array((0,1,0,1), dtype=bool)

def find_features_new(d, rs):
    features = []
    for region in rs:
        min_row, min_col, max_row, max_col = region.bbox
        if min_row <= 0 or min_col <= 0 or max_row >= d.shape[0] or max_col >= d.shape[1]:
            features.append(('edge', region.centroid))
            continue
        if max_row - min_row == 1 and max_col - min_col == 1:  # optimize 1-pixel case
            bounds = CELL
            edge = np.array((d[min_row-1][min_col], d[min_row][min_col+1], d[min_row+1][min_col], d[min_row][min_col-1])) > region.max_intensity
            if edge.all():
                features.append(('pit', region.centroid))
            elif (~edge).all():
                features.append(('peak', region.centroid))
            elif (edge == SADDLE_A).all() or (edge == SADDLE_B).all():
                features.append(('saddle', region.centroid))
        else:
            # can beat np.pad for our use case
            image = np.zeros(tuple(x+2 for x in region.image.shape), dtype='bool')
            image[1:-1:, 1:-1] = region.image
            dilated = scipy.ndimage.binary_dilation(image, structure=CROSS, border_value=0)
            bounds = dilated ^ image
            new_slices = tuple(slice(s.start-1,s.stop+1,s.step) for s in region.slice)
            data_im = d[new_slices].copy()
            g = data_im > region.max_intensity
            edge_greater = g[bounds]
            edge_less = ~edge_greater
            if edge_less.all():
                features.append(('peak', region.centroid))
            elif edge_greater.all():
                features.append(('pit', region.centroid))
            else:
                n_data = np.empty(data_im.shape, dtype='int')
                np.putmask(n_data, g, 1)
                np.putmask(n_data, ~g, 2)
                np.putmask(n_data, ~bounds, 0)
                groups, num = skimage.measure.label(n_data, connectivity=2, return_num=True, background=0)
                assert num > 1
                assert num % 2 == 0
                if num > 2:
                    features.append(('saddle', region.centroid))
    return features

# def work():
#     return find_features(data, padded, regions)

# import timeit

# timeit.timeit(work, number=1)

import line_profiler

def test(_slice):
    old_lp = line_profiler.LineProfiler()
    old_f = old_lp(find_features)
    old_res = old_f(data, padded, regions[_slice])
    new_lp = line_profiler.LineProfiler()
    new_f = new_lp(find_features_new)
    new_res = new_f(data[0].values, regions[_slice])
    assert old_res == new_res
    print('feature types:', set(f[0] for f in old_res))
    old_lp.print_stats()
    new_lp.print_stats()
