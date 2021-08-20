import matplotlib
from matplotlib import pyplot as plt

import rioxarray as rxr

import skimage
import skimage.measure
import skimage.color

def find_flats(d):
    return skimage.measure.label(d, connectivity=2)

import random

def random_cmap():
    colors = [(1,1,1)] + [(random.random(),random.random(),random.random()) for i in range(255)]
    return matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)

def region_info(d, l):
    return skimage.measure.regionprops(l, intensity_image=d)

import numpy as np

import scipy

ONE = np.ones((3,3), dtype=bool)

CELL = np.array(((1,1,1),(1,0,1),(1,1,1)), dtype=bool)

# SADDLE_B = np.array((0,1,0,1,0,1), dtype=bool)
# SADDLE_B = np.array

def find_features(d, rs):
    features = {}
    for region in rs:
        min_row, min_col, max_row, max_col = region.bbox
        if min_row <= 0 or min_col <= 0 or max_row >= d.shape[0] or max_col >= d.shape[1]:
            features[region.label] = 'edge'
            continue
        if max_row - min_row == 1 and max_col - min_col == 1:  # optimize 1-pixel case
            bounds = CELL
        else:
            # can beat np.pad for our use case
            image = np.zeros(tuple(x+2 for x in region.image.shape), dtype='bool')
            image[1:-1:, 1:-1] = region.image
            dilated = scipy.ndimage.binary_dilation(image, structure=ONE, border_value=0)
            bounds = dilated ^ image
        new_slices = tuple(slice(s.start-1,s.stop+1,s.step) for s in region.slice)
        data_im = d[new_slices].copy()
        g = data_im > region.max_intensity
        edge_greater = g[bounds]
        edge_less = ~edge_greater
        if edge_less.all():
            features[region.label] = 'peak'
        elif edge_greater.all():
            features[region.label] = 'pit'
        else:
            n_data = np.empty(data_im.shape, dtype='int')
            np.putmask(n_data, g, 1)
            np.putmask(n_data, ~g, 2)
            np.putmask(n_data, ~bounds, 0)
            groups, num = skimage.measure.label(n_data, connectivity=2, return_num=True, background=0)
            assert num > 1
            # assert num % 2 == 0
            if num > 2:
                features[region.label] = 'saddle'
    return features

def find_edge_highpoints(d, rs):
    edge_highpoints = np.full(len(rs)+1, -1)  # +1 since labels start at 1
    for region in rs:
        min_row, min_col, max_row, max_col = region.bbox
        if min_row <= 0 or min_col <= 0 or max_row >= d.shape[0] or max_col >= d.shape[1]:
            continue # edge
        if max_row - min_row == 1 and max_col - min_col == 1:  # optimize 1-pixel case
            bounds = CELL
        else:
            image = np.zeros(tuple(x+2 for x in region.image.shape), dtype='bool')
            image[1:-1:, 1:-1] = region.image
            dilated = scipy.ndimage.binary_dilation(image, structure=ONE, border_value=0)
            bounds = dilated ^ image
        new_slices = tuple(slice(s.start-1,s.stop+1,s.step) for s in region.slice)
        data_im = d[new_slices]
        edge = data_im * bounds
        r,c = np.unravel_index(edge.argmax(), edge.shape)
        edge_highpoints[region.label] = np.ravel_multi_index((min_row-1+r, min_col-1+c), d.shape)
    return edge_highpoints

def display_features(d, rs, features, radius=2):
    fig, ax = plt.subplots()
    ax.imshow(d)
    for region in rs:
        if region.label in features:
            ax.add_patch(matplotlib.patches.Circle(tuple(reversed(region.centroid)), radius=radius, color={'saddle': 'purple', 'peak': 'red', 'pit': 'blue', 'edge': 'white'}[features[region.label]]))
    plt.show()

def walk_up(start, labels, features, edge_highpoints):
    label = labels[start]
    if label in features and features[label] in {'edge', 'peak'}:
        return start
    return walk_up(np.unravel_index(edge_highpoints[label], labels.shape), labels, features, edge_highpoints)

def walk_up_draw(start, labels, features, edge_highpoints):
    xs, ys = [], []
    def rec(start):
        label = labels[start]
        if label in features and features[label] in {'edge', 'peak'}:
            return
        p = np.unravel_index(edge_highpoints[label], labels.shape)
        xs.append(p[1])
        ys.append(p[0])
        rec(p)
    xs.append(start[1])
    ys.append(start[0])
    rec(start)
    plt.plot(xs,ys)

def find_saddle_edge_highpoints(d, region):
    image = np.zeros(tuple(x+2 for x in region.image.shape), dtype='bool')
    image[1:-1:, 1:-1] = region.image
    dilated = scipy.ndimage.binary_dilation(image, structure=ONE, border_value=0)
    bounds = dilated ^ image
    new_slices = tuple(slice(s.start-1,s.stop+1,s.step) for s in region.slice)
    data_im = d[new_slices]
    n_data = np.empty(data_im.shape, dtype='int')
    g = data_im > region.max_intensity
    np.putmask(n_data, g, 1)
    np.putmask(n_data, ~g, 2)
    np.putmask(n_data, ~bounds, 0)
    groups, num = skimage.measure.label(n_data, connectivity=2, return_num=True, background=0)
    edge_regions = region_info(data_im, groups)
    saddle_edge_highpoints = []
    min_row, min_col, _, _ = region.bbox
    for edge_region in edge_regions:
        if edge_region.max_intensity > region.max_intensity:
            r,c = np.unravel_index(edge_region.intensity_image.argmax(), edge_region.intensity_image.shape)
            edge_min_row, edge_min_col, _, _ = edge_region.bbox
            saddle_edge_highpoints.append((min_row+edge_min_row+r-1,min_col+edge_min_col+c-1))
    return saddle_edge_highpoints

def draw_from_saddles(d, rs, labels, features, edge_highpoints):
    mask = np.ma.masked_all_like(d)
    fig, ax = plt.subplots()
    ax.imshow(d)
    for label,t in features.items():
        region = rs[label-1]
        if t in {'saddle', 'peak'}:
            for r,c in region.coords:  # stupid
                mask[r,c] = 0 if t == 'saddle' else 1
        if t == 'saddle':
            for hp in find_saddle_edge_highpoints(d, region):
                walk_up_draw(hp, labels, features, edge_highpoints)
    ax.imshow(mask, cmap='hot')
    plt.show()

import networkx as nx

import itertools

def build_divide_tree(d, rs, labels, features, edge_highpoints):
    tree = nx.Graph()
    for label,t in features.items():
        region = rs[label-1]
        if t == 'saddle':
            peaks = []
            for hp in find_saddle_edge_highpoints(d, region):
                end_coords = walk_up(hp, labels, features, edge_highpoints)
                end_label = labels[end_coords]
                end_t = features[end_label]
                if end_t == 'peak':
                    peaks.append(end_label)
            for p1,p2 in itertools.combinations(peaks, r=2):
                if p1 != p2:
                    if tree.has_node(p1) and tree.has_node(p2) and nx.has_path(tree, p1, p2):
                        path = nx.shortest_path(tree, p1, p2)
                        edges = list(zip(path[:-1], path[1:]))
                        basin_saddle = min(edges, key=lambda e: tree.edges[e]['elevation'])
                        tree.remove_edge(*basin_saddle)
                    tree.add_edge(p1, p2, saddle=label, elevation=region.mean_intensity)
    return tree

def draw_divide_tree(d, rs, tree):
    plt.imshow(d)
    for p1,p2 in tree.edges():
        p1_c = rs[p1-1].centroid
        p2_c = rs[p2-1].centroid
        saddle_c = rs[tree.edges[p1,p2]['saddle']-1].centroid
        plt.plot((p1_c[1],saddle_c[1],p2_c[1]), (p1_c[0],saddle_c[0],p2_c[0]))
    plt.show()

import time

def time_call(f, *args, **kwargs):
    start = time.perf_counter()
    res = f(*args, **kwargs)
    print(f'{f.__name__} took {time.perf_counter() - start:.2f}s')
    return res

import line_profiler

def profile_call(f, *args, **kwargs):
    lp = line_profiler.LineProfiler()
    f_ = lp(f)
    res = f_(*args, **kwargs)
    lp.print_stats()
    return res

def load_and_draw_divide_tree(filename):
    d = time_call(rxr.open_rasterio, filename, masked=True)
    d_n = d[0].values
    labels = time_call(find_flats, d_n)
    regions = time_call(region_info, d_n, labels)
    features = time_call(find_features, d_n, regions)
    edge_highpoints = time_call(find_edge_highpoints, d_n, regions)
    tree = time_call(build_divide_tree, d_n, regions, labels, features, edge_highpoints)
    draw_divide_tree(d_n, regions, tree)
