from __future__ import division
import os
from glob import glob
from collections import defaultdict
from math import pi, degrees, radians, atan2, sqrt, log, acos
from random import (uniform,
                    sample,
                    choice,
                   )
from string import ascii_lowercase
from itertools import tee, izip, product, combinations_with_replacement
import multiprocessing
import numpy as np
from scipy import ndimage as ndi
from scipy.misc import imread
from scipy.signal import find_peaks_cwt
from scipy.spatial.distance import euclidean, pdist
from scipy.ndimage.interpolation import rotate, shift
from scipy.ndimage.filters import median_filter, gaussian_filter1d
from scipy.ndimage.morphology import binary_fill_holes
from scipy.stats import norm
from skimage import draw
from skimage import io
from skimage.color import (rgb2gray,
                           label2rgb,
                           rgb2lab,
                           rgb2hsv,
                           rgb2xyz,
                           rgb2luv,
                          )
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from skimage.morphology import (watershed,
                                reconstruction,
                                erosion,
                                disk,
                                rectangle,
                                opening,
                                closing,
                                skeletonize,
                                binary_opening,
                                binary_closing,
                                binary_dilation,
                                binary_erosion,
                               )
from skimage.transform import probabilistic_hough_line, rescale, resize
from skimage.feature import (peak_local_max,
                             hessian_matrix,
                             blob_log,
                            )
from skimage.filters import (threshold_otsu,
                             gaussian,
                             sobel,
                             threshold_local,
                             median,
                            )
from skimage.util import invert
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import networkx as nx
import shapely
from shapely import geometry
from plotly.offline import (download_plotlyjs,
                            init_notebook_mode,
                            iplot,
                           )
from plotly import graph_objs
init_notebook_mode()


def epoch_to_hash(epoch):
    """
    Generate an alphanumeric hash from a Unix epoch. Unix epoch is
    rounded to the nearest second before hashing.
    Arguments:
        epoch: Unix epoch time. Must be positive.
    Returns:
        Alphanumeric hash of the Unix epoch time.
    Cribbed from Scott W Harden's website
    http://www.swharden.com/blog/2014-04-19-epoch-timestamp-hashing/
    """
    if epoch <= 0:
        raise ValueError("epoch must be positive.")
    epoch = round(epoch)
    hashchars = '0123456789abcdefghijklmnopqrstuvwxyz'
    #hashchars = '01' #for binary
    epoch_hash = ''
    while epoch > 0:
        epoch_hash = hashchars[int(epoch % len(hashchars))] + epoch_hash
        epoch = int(epoch / len(hashchars))
    return epoch_hash


class Plate(object):
    def __init__(self,
                 image,
                 tag_in='original_image',
                 source_filename=None,
                ):
        self.image_stash = {tag_in: image.copy()}
        self.feature_stash = {}
        self.metadata = {'source_filename': source_filename}

    def invert_image(self,
                     tag_in,
                     tag_out='inverted_image',
                    ):
        self.image_stash[tag_out] = invert(self.image_stash[tag_in])
        return self.image_stash[tag_out], None

    def crop_to_plate(self,
                      tag_in,
                      tag_out,
                      feature_out='crop_rotation',
                     ):
        image = self.image_stash[tag_in]
        g_img = rgb2gray(image)
        t_img = (g_img > threshold_otsu(g_img)).astype(np.uint8)
        labels, num_labels = (ndi
                              .measurements
                              .label(ndi.binary_fill_holes(t_img))
                             )
        objects = ndi.measurements.find_objects(labels)
        largest_object = max(objects, key=lambda x:image[x].size)
        rp = regionprops(t_img, intensity_image=t_img)
        assert len(rp) == 1
        rads = rp[0].orientation
        rads %= 2 * pi
        rotation = 90 - degrees(rads)
        r_img = rotate(image[largest_object],
                       angle=rotation,
                       axes=(1, 0),
                       reshape=True,
                       output=None,
                       order=5,
                       mode='constant',
                       cval=0.0,
                       prefilter=True,
                      )
        gr_img = rgb2gray(r_img)
        tr_img = (gr_img > threshold_otsu(gr_img)).astype(np.uint8)
        labels, num_labels = (ndi
                              .measurements
                              .label(ndi.binary_fill_holes(tr_img))
                             )
        objects = ndi.measurements.find_objects(labels)
        largest_object = max(objects, key=lambda x:r_img[x].size)
        self.image_stash[tag_out] = r_img[largest_object]
        self.feature_stash[feature_out] = rotation
        return self.image_stash[tag_out], self.feature_stash[feature_out]

    def crop_border(self,
                    tag_in,
                    tag_out='cropped_image',
                    border=30,
                   ):
        if border <= 0:
            raise ValueError("border must be > 0")
        if min(self.image_stash[tag_in].shape[:2]) < 2 * border + 1:
            raise ValueError("Cannot crop to image with 0 pixels.")
        cropped_image = self.image_stash[tag_in][border:-border,border:-border]
        self.image_stash[tag_out] = cropped_image
        return self.image_stash[tag_out], None

    @staticmethod
    def extend_line(line, image):
        #In image space x is width, y is height
        #(w1, h1), (w2, h2) = (x1, y1), (x2, y2)
        (w1, h1), (w2, h2) = line
        image_height, image_width = image.shape[:2]
        ihm, iwm = image_height - 1, image_width - 1
        if h1 == h2:
            left_width = 0
            left_height = h1
            right_width = iwm
            right_height = h1
            extended_line = ((left_width, left_height),
                             (right_width, right_height))
        elif w1 == w2:
            left_width = w1
            left_height = 0
            right_width = w1
            right_height = ihm
            extended_line = ((left_width, left_height),
                             (right_width, right_height))
        else:
            slope_h = (w2 - w1) / (h2 - h1)
            slope_w = (h2 - h1) / (w2 - w1)
            top_border_intersection = (0, w1 - slope_h * h1)
            right_border_intersection = (h1 + slope_w * (iwm - w1), iwm)
            bottom_border_intersection = (ihm, w1 + slope_h * (ihm - h1))
            left_border_intersection = (h1 - slope_w * w1, 0)
            border_intersections = (top_border_intersection,
                                    right_border_intersection,
                                    bottom_border_intersection,
                                    left_border_intersection,
                                   )
            valid_borders = []
            for BH, BW in border_intersections:
                if 0 <= BH <= ihm and 0 <= BW <= iwm:
                    valid_borders.append(True)
                else:
                    valid_borders.append(False)
            bounding_points = []
            for i, intersection in enumerate(border_intersections):
                if valid_borders[i]:
                    bounding_points.append(intersection)
            assert len(bounding_points) > 1
            bounding_points = bounding_points[:2]
            bounding_points = sorted(bounding_points, key=lambda x:x[0])
            ((left_height, left_width),
             (right_height, right_width)) = bounding_points
            extended_line = ((left_width, left_height),
                             (right_width, right_height))
        return extended_line

    def find_baseline(self,
                      tag_in,
                      feature_out='baseline',
                     ):
        g_img = rgb2gray(self.image_stash[tag_in])
        height, width = g_img.shape
        gg_img = gaussian(g_img, sigma=1)
        sg_img = sobel(gg_img)
        tsg_img = sg_img > threshold_otsu(sg_img)
        lines = probabilistic_hough_line(tsg_img,
                                         line_length=width * 0.60,
                                         line_gap=width * 0.60 / 10.0)
        #Select for lines that are approximately horizontal.
        angle_filtered_lines = []
        for line in lines:
            (x1, y1), (x2, y2) = line
            if x1 == x2:
                continue
            rads = atan2(abs(y1 - y2), abs(x1 - x2))
            rads %= 2 * pi
            degs = degrees(rads)
            if -20 < degs < 20 or 160 < degs < 200:
                angle_filtered_lines.append(line)
        final_line = max(angle_filtered_lines, key=euclidean)
        self.feature_stash[feature_out] = Plate.extend_line(final_line, g_img)
        return None, self.feature_stash[feature_out]

    def baseline_orient(self,
                        tag_in,
                        tag_out='baseline_oriented_image',
                        baseline_feature='baseline',
                        feature_out='reoriented_baseline',
                       ):
        baseline = self.feature_stash[baseline_feature]
        (x1, y1), (x2, y2) = baseline
        image_height = self.image_stash[tag_in]
        if min(y1, y2) < abs(max(y1, y2) - image_height):
            reoriented_image = np.fliplr(np.flipud(self.image_stash[tag_in]))
            self.image_stash[tag_out] = reoriented_image
            reoriented_baseline = ((x1, image_height - y2),
                                   (x2, image_height - y1))
            self.feature_stash[feature_out] = reoriented_baseline
        else:
            #No need to reorient
            self.image_stash[tag_out] = self.image_stash[tag_in].copy()
            self.feature_stash[feature_out] = baseline
        return self.image_stash[tag_out], self.feature_stash[feature_out]

    def baseline_mean(self,
                      baseline_feature='baseline',
                     ):
        baseline = self.feature_stash[baseline_feature]
        (w1, h1), (w2, h2) = (x1, y1), (x2, y2) = baseline
        mean_h, mean_w = np.mean((h1, h2)), np.mean((w1, w2))
        return mean_h, mean_w

    def display(self,
                tag_in,
                figsize=20,
                basins_feature=None,
                basin_alpha=0.1,
                baseline_feature=None,
                solvent_front_feature=None,
                lanes_feature=None,
                basin_centroids_feature=None,
                basin_lane_assignments_feature=None,
                basin_intensities_feature=None,
                basin_rfs_feature=None,
                lines_feature=None,
                draw_boundaries=True,
                side_by_side=False,
                display_labels=False,
                text_color='black',
                fontsize='12',
                blobs_feature=None,
                output_filename=None
               ):
        image_shown = self.image_stash[tag_in]
        image_height, image_width = image_shown.shape[0], image_shown.shape[1]
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(figsize, figsize))
        basins = (None if basins_feature is None
                  else self.feature_stash[basins_feature])
        if basins is None:
            ax.imshow(image_shown)
        else:
            g_img = rgb2gray(image_shown)
            if draw_boundaries:
                boundaries = find_boundaries(basins, mode='inner')
                bg_img = g_img * ~boundaries
            else:
                bg_img = g_img
            ax.imshow(label2rgb(basins, image=bg_img, alpha=basin_alpha))
        baseline = (None if baseline_feature is None
                    else self.feature_stash[baseline_feature])
        if baseline is not None:
            (w1, h1), (w2, h2) = (x1, y1), (x2, y2) = baseline
            ax.plot((w1, w2),
                    (h1, h2),
                    color='orange',
                    linestyle='-',
                    linewidth=1,
                   )
        lines = (None if lines_feature is None
                 else self.feature_stash[lines_feature])
        if lines is not None:
            for line in lines:
                (w1, h1), (w2, h2) = (x1, y1), (x2, y2) = line
                ax.plot((w1, w2),
                        (h1, h2),
                        color='yellow',
                        linestyle='-',
                        linewidth=1,
                        )
        solvent_front = (None if solvent_front_feature is None
                         else self.feature_stash[solvent_front_feature])
        if solvent_front is not None:
            (w1, h1), (w2, h2) = (x1, y1), (x2, y2) = solvent_front
            ax.plot((w1, w2),
                    (h1, h2),
                    color='purple',
                    linestyle='-',
                    linewidth=1,
                   )
        lanes = (None if lanes_feature is None
                 else self.feature_stash[lanes_feature])
        if lanes is not None:
            for lane in lanes:
                draw_h = (0, image_height)
                draw_w = (lane, lane)
                ax.plot(draw_w,
                        draw_h,
                        color='green',
                        linestyle='-',
                        linewidth=1,
                       )
        basin_centroids = (None if basin_centroids_feature is None
                           else self.feature_stash[basin_centroids_feature])
        basin_lane_assignments = (
                      None if basin_lane_assignments_feature is None
                      else self.feature_stash[basin_lane_assignments_feature]
                                 )
        basin_intensities = (
                           None if basin_intensities_feature is None
                           else self.feature_stash[basin_intensities_feature]
                            )
        basin_rfs = (None if basin_rfs_feature is None
                     else self.feature_stash[basin_rfs_feature])
        if basin_centroids is not None:
            for Label, centroid in basin_centroids.iteritems():
                x, y = centroid
                if display_labels:
                    display_text = str(Label)
                else:
                    display_text = ''
                if basin_lane_assignments is not None:
                    display_text += ("row " +
                                     str(basin_lane_assignments[Label]))
                if basin_intensities is not None:
                    display_text += ("; I = " +
                                     str(basin_intensities[Label]))
                if basin_rfs is not None and Label in basin_rfs:
                    display_text += ("; rf = " +
                                     str(round(basin_rfs[Label], 2)))
                plt.text(y, x,
                         display_text,
                         color=text_color,
                         fontsize=fontsize,
                        )
        if blobs_feature is not None:
            blobs = self.feature_stash[blobs_feature]
            for blob in blobs:
                y, x, r = blob
                c = plt.Circle((x, y),
                               r,
                               color='red',
                               linewidth=2,
                               fill=False,
                              )
                ax.add_patch(c)
        if output_filename is None:
            plt.show()
        else:
            plt.savefig(output_filename)
            plt.close(fig)
        if side_by_side:
            #Currently, this is not really side-by-side, but below.
            fig, ax = plt.subplots(ncols=1, nrows=1,
                                   figsize=(figsize, figsize))
            ax.imshow(image_shown)
            if baseline is not None:
                (w1, h1), (w2, h2) = (x1, y1), (x2, y2) = baseline
                ax.plot((w1, w2),
                        (h1, h2),
                        color='orange',
                        linestyle='-',
                        linewidth=1,
                       )
            if lanes is not None:
                for lane in lanes:
                    draw_h = (0, image_height)
                    draw_w = (lane, lane)
                    ax.plot(draw_w,
                            draw_h,
                            color='green',
                            linestyle='-',
                            linewidth=1,
                           )
            if solvent_front is not None:
                (w1, h1), (w2, h2) = (x1, y1), (x2, y2) = solvent_front
                ax.plot((w1, w2),
                        (h1, h2),
                        color='purple',
                        linestyle='-',
                        linewidth=1,
                       )
            if output_filename is None:
                plt.show()
            else:
                plt.savefig(output_filename)
                plt.close(fig)

    @staticmethod
    def get_baseline_H_domain(baseline,
                              baseline_radius,
                             ):
        (LW, LH), (RW, RH) = baseline
        min_H, max_H = min(LH, RH), max(LH, RH)
        lower_H = max(0, min_H - baseline_radius)
        upper_H = max_H + baseline_radius + 1
        return lower_H, upper_H

    @staticmethod
    def find_max_rp_coord(rp):
        min_row, min_col, max_row, max_col = rp.bbox
        hm, wm = np.unravel_index(np.argmax(rp.intensity_image),
                                  rp.intensity_image.shape)
        return min_row + hm, min_col + wm

    @staticmethod
    def rp_intensity(rp,
                     background,
                     background_basins=None,
                     radius=None,
                     radius_factor=None,
                     negative=False,
                     multiplier=1,
                    ):
        min_row, min_col, max_row, max_col = rp.bbox
        if radius is None:
            radius = int(np.ceil((sqrt(2) - 1) *
                                 max(abs(max_row - min_row),
                                     abs(max_col - min_col))))
        if radius_factor is not None:
            radius *= radius_factor
            radius = int(np.ceil(radius))
        subimage_minh = max(0, min_row - radius)
        subimage_maxh = max_row + radius
        subimage_minw = max(0, min_col - radius)
        subimage_maxw = max_col + radius
        subimage = background[subimage_minh:subimage_maxh,
                              subimage_minw:subimage_maxw]
        if background_basins is None:
            median_correction = np.median(subimage)
        else:
            subimage_basins =  background_basins[subimage_minh:subimage_maxh,
                                                 subimage_minw:subimage_maxw]
            filtered_subimage = subimage[np.where(subimage_basins == 0)]
            if len(filtered_subimage) == 0:
                median_correction = np.median(subimage)
            else:
                median_correction = np.median(filtered_subimage)
        intensity = (rp.mean_intensity - median_correction) * rp.area
        if negative:
            intensity *= -1
        intensity *= multiplier
        return intensity

    @staticmethod
    def pairwise(iterable):
        """
        Produces an iterable that yields "s -> (s0, s1), (s1, s2), (s2, s3)..."
        From Python itertools recipies.

        e.g.

        a = pairwise([5, 7, 11, 4, 5])
        for v, w in a:
            print [v, w]

        will produce

        [5, 7]
        [7, 11]
        [11, 4]
        [4, 5]

        The name of this function reminds me of Pennywise.
        """
        a, b = tee(iterable)
        next(b, None)
        return izip(a, b)

    def find_notches(self,
                     tag_in,
                     feature_out='notches',
                     baseline_feature='baseline',
                     baseline_radius=20,
                     sigma=10,
                    ):
        baseline = self.feature_stash[baseline_feature]
        lower_H, upper_H = Plate.get_baseline_H_domain(baseline,
                                                       baseline_radius)
        g_img = rgb2gray(self.image_stash[tag_in])
        baseline_slice = g_img[lower_H:upper_H,:]
        col_sum = np.sum(baseline_slice, axis=0)
        grs = gaussian_filter1d(col_sum, sigma=sigma)
        peaks = find_peaks_cwt(np.amax(grs) - grs, np.arange(5, 30))
        self.feature_stash[feature_out] = sorted(peaks)
        return None, self.feature_stash[feature_out]

    def lanes_from_notches(self,
                           tag_in,
                           notches_feature='notches',
                           feature_out='lanes',
                          ):
        notches = self.feature_stash[notches_feature]
        midpoints = [np.mean([n, notches[i]])
                     for i, n in enumerate(notches[1:])]
        image_width = self.image_stash[tag_in].shape[1]
        lanes = [0] + midpoints + [image_width]
        self.feature_stash[feature_out] = lanes
        return None, self.feature_stash[feature_out]

    def find_basin_centroids(self,
                             tag_in,
                             basins_feature='basins',
                             feature_out='basin_centroids',
                            ):
        intensity_image = rgb2gray(self.image_stash[tag_in])
        basins = self.feature_stash[basins_feature]
        RP = regionprops(label_image=basins, intensity_image=intensity_image)
        basin_centroids = {rp.label: rp.centroid for rp in RP}
        self.feature_stash[feature_out] = basin_centroids
        return None, self.feature_stash[feature_out]

    def lane_assign_basins(self,
                           lanes_feature='lanes',
                           basin_centroids_feature='basin_centroids',
                           feature_out='lane_assignments',
                          ):
        lanes = self.feature_stash[lanes_feature]
        if len(lanes) > len(ascii_lowercase):
            raise NotImplementedError("Does not yet handle more lanes than "
                                      "letters in the English alphabet.")
        basin_centroids = self.feature_stash[basin_centroids_feature]
        lane_assignments = {}
        for Label, centroid in basin_centroids.iteritems():
            x, y = centroid
            for i, (llane, rlane) in enumerate(Plate.pairwise(lanes)):
                if llane <= y < rlane:
                    lane_assignments[Label] = ascii_lowercase[i]
            else:
                lane_assignments[Label] = 'U'
        self.feature_stash[feature_out] = lane_assignments
        return None, self.feature_stash[feature_out]

    def measure_basin_intensities(self,
                                  tag_in,
                                  median_radius=None,
                                  filter_basins=False,
                                  radius_factor=None,
                                  basins_feature='basins',
                                  feature_out='basin_intensities',
                                  multiplier=1,
                                 ):
        g_img = rgb2gray(self.image_stash[tag_in])
        if median_radius is not None:
            mg_img = median(g_img, selem=disk(median_radius))
        else:
            mg_img = g_img
        basins = self.feature_stash[basins_feature]
        RP = regionprops(label_image=basins, intensity_image=g_img)
        if filter_basins:
            background_basins = basins
        else:
            background_basins = None
        basin_intensities = {rp.label:
                             int(round(Plate.rp_intensity(
                                           rp=rp,
                                           background=mg_img,
                                           background_basins=background_basins,
                                           radius=None,
                                           radius_factor=radius_factor,
                                           negative=True,
                                           multiplier=multiplier,
                                                         )
                                      )
                                )
                             for rp in RP}
        #TODO: Subtract notch intensities from blobs near baseline.
        self.feature_stash[feature_out] = basin_intensities
        return None, self.feature_stash[feature_out]

    def find_solvent_front(self,
                           tag_in,
                           sigma=4,
                           feature_out='solvent_front',
                          ):
        g_img = rgb2gray(self.image_stash[tag_in])
        g_img = g_img / np.amax(g_img)
        scale = 100.0 / g_img.shape[0]
        rg_img = rescale(g_img, scale=scale)
        Hxx, Hxy, Hyy = hessian_matrix(rg_img, sigma=sigma)
        row_mean = np.mean(Hyy, axis=1)
        row_std = np.std(Hyy, axis=1)
        row_sm = [float(x) / float(row_std[i]) for i, x in enumerate(row_mean)]
        sf = np.argmax(row_sm)
        if row_sm[sf] < 1:
            sf = 0
        else:
            sf /= scale
        solvent_front = sf
        self.feature_stash[feature_out] = solvent_front
        return None, self.feature_stash[feature_out]

    @staticmethod
    def translate_line(line,
                       h, w,
                       extend=True,
                       image=None,
                      ):
        (w1, h1), (w2, h2) = line
        translated_line = (w1 + w, h1 + h), (w2 + w, h2 + h)
        if extend:
            if image is None:
                raise ValueError("If extending, need image.")
            translated_line = Plate.extend_line(line=translated_line,
                                                image=image,
                                               )
        return translated_line

    def compute_basin_rfs(self,
                          basin_centroids_feature='basin_centroids',
                          baseline_feature='baseline',
                          solvent_front_feature='solvent_front',
                          feature_out='basin_rfs',
                         ):
        basin_centroids = self.feature_stash[basin_centroids_feature]
        baseline = self.feature_stash[baseline_feature]
        solvent_front = self.feature_stash[solvent_front_feature]
        basin_rfs = {}
        for Label, centroid in basin_centroids.iteritems():
            distance_to_base = Plate.point_line_distance(point=centroid[::-1],
                                                         line=baseline,
                                                        )
            distance_to_front = Plate.point_line_distance(point=centroid[::-1],
                                                          line=solvent_front,
                                                         )
            base_P1, base_P2 = baseline
            front_P1, front_P2 = solvent_front
            cb_1, cb_2 = (centroid[::-1], base_P1), (centroid[::-1], base_P2)
            cf_1, cf_2 = (centroid[::-1], front_P1), (centroid[::-1], front_P2)
            intersects_front_1 = Plate.line_segments_intersect(
                                                       segment_A=cb_1,
                                                       segment_B=solvent_front,
                                                              )
            intersects_front_2 = Plate.line_segments_intersect(
                                                       segment_A=cb_2,
                                                       segment_B=solvent_front,
                                                              )
            intersects_front = intersects_front_1 or intersects_front_2
            intersects_base_1 = Plate.line_segments_intersect(
                                                            segment_A=cf_1,
                                                            segment_B=baseline,
                                                             )
            intersects_base_2 = Plate.line_segments_intersect(
                                                            segment_A=cf_2,
                                                            segment_B=baseline,
                                                             )
            intersects_base = intersects_base_1 or intersects_base_2
            assert not (intersects_front and intersects_base)
            if intersects_front:
                denominator = distance_to_base - distance_to_front
                assert denominator > 0, (distance_to_base,
                                         distance_to_front,
                                         centroid,
                                         'intersects_front',
                                        )
                rf = distance_to_base / denominator
            elif intersects_base:
                denominator = distance_to_front - distance_to_base
                assert denominator > 0, (distance_to_base,
                                         distance_to_front,
                                         centroid,
                                         'intersects_base',
                                        )
                rf = -distance_to_base / denominator
            else:
                denominator = distance_to_front + distance_to_base
                assert denominator > 0, (distance_to_base,
                                         distance_to_front,
                                         centroid,
                                         'neither front nor base',
                                        )
                rf = distance_to_base / denominator
            basin_rfs[Label] = rf
        #baseline_mean = self.baseline_mean(baseline_feature=baseline_feature)
        #if solvent_front == 0 or solvent_front == baseline_mean:
        #    basin_rfs = None
        #else:
        #    basin_rfs = {}
        #    for Label, centroid in basin_centroids.iteritems():
        #        x, y = centroid
        #        if baseline_mean != solvent_front:
        #            rf = ((baseline_mean - x) /
        #                  (baseline_mean - solvent_front))
        #        else:
        #            rf = 'NaN'
        #        basin_rfs[Label] = rf
        self.feature_stash[feature_out] = basin_rfs
        return None, self.feature_stash[feature_out]

    @staticmethod
    def median_correct_image(image,
                             median_disk_radius,
                            ):
        g_img = rgb2gray(image)
        if median_disk_radius is None or median_disk_radius == 0:
            mg_img = g_img.copy()
        else:
            m_img = median(g_img, selem=disk(median_disk_radius))
            mg_img = g_img * np.mean(m_img) / m_img
        return mg_img

    @staticmethod
    def make_bT_bF(image, dtype=np.bool):
        bT = np.ones_like(image, dtype=dtype)
        bF = np.zeros_like(image, dtype=dtype)
        return bT, bF

    @staticmethod
    def open_close_boolean_basins(boolean_basins,
                                  open_close_size=10,
                                 ):
        open_close_disk = disk(open_close_size)
        opened = binary_opening(boolean_basins, selem=open_close_disk)
        closed_opened = binary_closing(opened, selem=open_close_disk)
        return closed_opened

    @staticmethod
    def open_close_label_basins(basins,
                                open_close_size=10,
                                exclude_label=None,
                                exclude_labels=None,
                               ):
        """Excluded labels are set to 0."""
        if exclude_labels is not None:
            exclude_labels = set(exclude_labels)
        else:
            exclude_labels = set()
        if exclude_label is not None:
            exclude_labels.add(exclude_label)
        open_closed_basins = np.zeros_like(basins)
        bT, bF = Plate.make_bT_bF(image=basins, dtype=np.bool)
        for L in np.unique(basins):
            if L in exclude_labels:
                continue
            label_boolean = np.where(basins == L, bT, bF)
            open_closed = Plate.open_close_boolean_basins(
                                               boolean_basins=label_boolean,
                                               open_close_size=open_close_size,
                                                         )
            open_closed_basins = np.where(open_closed, L, open_closed_basins)
        open_closed_basins = label(open_closed_basins)
        return open_closed_basins

    @staticmethod
    def erode_basins(basins,
                     erosion_size=10,
                     rectangular=False,
                     exclude_label=None,
                     exclude_labels=None,
                    ):
        """Excluded labels are set to 0."""
        if exclude_labels is not None:
            exclude_labels = set(exclude_labels)
        else:
            exclude_labels = set()
        if exclude_label is not None:
            exclude_labels.add(exclude_label)
        eroded_basins = np.zeros_like(basins)
        bT, bF = Plate.make_bT_bF(image=basins, dtype=np.bool)
        if rectangular:
            side_length = 2 * erosion_size + 1
            erosion_element = rectangle(width=side_length, height=side_length)
        else:
            erosion_element = disk(erosion_size)
        for L in np.unique(basins):
            if L in exclude_labels:
                continue
            label_boolean = np.where(basins == L, bT, bF)
            eroded = binary_erosion(image=label_boolean, selem=erosion_element)
            eroded_basins = np.where(eroded, L, eroded_basins)
        eroded_basins = label(eroded_basins)
        return eroded_basins

    @staticmethod
    def dilate_basins(basins,
                      dilation_size=10,
                      rectangular=False,
                      exclude_label=None,
                      exclude_labels=None,
                     ):
        """Excluded labels are set to 0."""
        if exclude_labels is not None:
            exclude_labels = set(exclude_labels)
        else:
            exclude_labels = set()
        if exclude_label is not None:
            exclude_labels.add(exclude_label)
        dilated_basins = np.zeros_like(basins)
        bT, bF = Plate.make_bT_bF(image=basins, dtype=np.bool)
        if rectangular:
            side_length = 2 * dilation_size + 1
            dilation_element = rectangle(width=side_length, height=side_length)
        else:
            dilation_element = disk(dilation_size)
        for L in np.unique(basins):
            if L in exclude_labels:
                continue
            label_boolean = np.where(basins == L, bT, bF)
            dilated = binary_dilation(image=label_boolean,
                                      selem=dilation_element)
            dilated_basins = np.where(dilated, L, dilated_basins)
        dilated_basins = label(dilated_basins)
        return dilated_basins

    @staticmethod
    def skeletonize_basins(basins,
                           exclude_label=None,
                           exclude_labels=None,
                          ):
        if exclude_labels is not None:
            exclude_labels = set(exclude_labels)
        else:
            exclude_labels = set()
        if exclude_label is not None:
            exclude_labels.add(exclude_label)
        skeletonized_basins = np.zeros_like(basins)
        for L in np.unique(basins):
            if L in exclude_labels:
                continue
            label_boolean = np.where(basins == L, True, False)
            skeletonized = skeletonize(image=label_boolean)
            skeletonized_basins = np.where(skeletonized,
                                           L,
                                           skeletonized_basins,
                                          )
        skeletonized_basins = label(skeletonized_basins)
        return skeletonized_basins

    @staticmethod
    def most_frequent_label(basins,
                            image=None,
                           ):
        label_counts = np.bincount(basins.reshape(-1))
        most_frequent_label = np.argmax(label_counts)
        background_pixel_coordinates = np.where(basins == most_frequent_label)
        if image is not None:
            background_pixel_values = image[background_pixel_coordinates]
        else:
            background_pixel_values = None
        return (most_frequent_label,
                background_pixel_coordinates,
                background_pixel_values,
               )

    def remove_most_frequent_label(self,
                                   basins_feature='basins',
                                   feature_out='filtered_basins',
                                   debug_output=False,
                                  ):
        basins = self.feature_stash[basins_feature]
        (most_frequent_label,
         background_pixel_coordinates,
         background_pixel_values,
        ) = Plate.most_frequent_label(basins=basins)
        empty_basin_template = np.zeros_like(basins)
        filtered_basins = np.where(basins != most_frequent_label,
                                   basins,
                                   empty_basin_template,
                                  )
        filtered_basins = label(filtered_basins)
        #if debug_output:
        #    print("filtered basins debug")
        #    light_dummy = np.ones_like(basins) * np.iinfo(basins.dtype).max
        #    self.image_stash['debug_display'] = light_dummy
        #    self.feature_stash['debug_basins'] = filtered_basins
        #    self.display(tag_in='debug_display',
        #                 basins_feature='debug_basins',
        #                 figsize=10,
        #                 display_labels=True,
        #                )
        self.feature_stash[feature_out] = filtered_basins
        return None, self.feature_stash[feature_out]

    def waterfall_segmentation(self,
                               tag_in,
                               feature_out='waterfall_basins',
                               R_out='R_img',
                               mg_out='mg_img',
                               median_disk_radius=31,
                               smoothing_sigma=0,
                               threshold_opening_size=3,
                               basin_open_close_size=10,
                               skeleton_label=0,
                               debug_output=False,
                              ):
        """
        Algorithm based on

        Beucher, Serge. "Watershed, hierarchical segmentation and waterfall
        algorithm." Mathematical morphology and its applications to image
        processing. Springer Netherlands, 1994. 69-76.
        DOI 10.1007/978-94-011-1040-2_10
        """
        working_image = self.image_stash[tag_in]
        if debug_output:
            print("waterfall input image debug")
            self.image_stash['debug_display'] = working_image
            self.display(tag_in='debug_display',
                         figsize=10,
                        )
        o_img = working_image
        g_img = rgb2gray(o_img)
        if smoothing_sigma > 0:
            g_img = gaussian(g_img, sigma=smoothing_sigma)
        if debug_output:
            print("smoothing image debug")
            self.image_stash['debug_display'] = g_img
            self.display(tag_in='debug_display',
                         figsize=10,
                        )
        if median_disk_radius is None:
            median_disk_radius = (max(g_img.shape) // 2) * 2 + 1
            mg_img = g_img.copy()
        else:
            mg_img = \
              Plate.median_correct_image(image=g_img,
                                         median_disk_radius=median_disk_radius)
        self.image_stash[mg_out] = mg_img.copy()
        if debug_output:
            print("median debug")
            self.image_stash['debug_display'] = mg_img
            self.display(tag_in='debug_display',
                         figsize=10,
                        )
        #find maxima at high resolution
        n_img = np.amax(mg_img) - mg_img
        maxima_distance = 5 #using 'thick' boundaries below,
                            #so this needs to be sane
        local_maxima = peak_local_max(n_img, indices=False,
                                      min_distance=maxima_distance)
        markers = label(local_maxima)
        #perform watershed
        W_labels = watershed(mg_img, markers=markers)
        #find boundaries, which is the actual W
        W = find_boundaries(W_labels, connectivity=1, mode='thick')
        mg_max = np.amax(mg_img)
        mg_max_array = np.ones_like(mg_img)
        mg_max_array *= mg_max
        g = np.where(W, mg_img, mg_max_array)
        if debug_output:
            print("g debug")
            self.image_stash['debug_display'] = g
            self.display(tag_in='debug_display',
                         figsize=10,
                        )
        #reconstruction by erosion
        R = reconstruction(g, mg_img, method='erosion')
        self.image_stash[R_out] = R.copy()
        if debug_output:
            print("R debug")
            self.image_stash['debug_display'] = R
            self.display(tag_in='debug_display',
                         figsize=10,
                        )
        #perform watershed on R
        n_R = np.amax(R) - R
        #thresh = threshold_li(n_R)
        #thresh = threshold_otsu(n_R)
        thresh = threshold_local(n_R, median_disk_radius)
        thresh_image = n_R > thresh
        thresh_image = opening(image=thresh_image,
                               selem=disk(threshold_opening_size))
        if debug_output:
            print("thresh_image debug")
            self.image_stash['debug_display'] = thresh_image
            self.display(tag_in='debug_display',
                         figsize=10,
                        )
        local_maxi = thresh_image
        #so that local_maxi and skel don't overlap
        local_maxi_compliment = erosion(~local_maxi, disk(2))
        skel = skeletonize(local_maxi_compliment)
        skel = np.logical_xor(skel, np.logical_and(skel, local_maxi))
        local_maxi = np.logical_or(local_maxi, skel)
        markers = label(local_maxi)
        if debug_output:
            print("local_maxi markers debug")
            self.image_stash['debug_display'] = g_img
            self.feature_stash['debug_basins'] = markers
            self.display(tag_in='debug_display',
                         basins_feature='debug_basins',
                         figsize=10,
                         display_labels=True,
                        )
        WR_labels = watershed(R, markers=markers)
        if skeleton_label is not None:
            superlabel = np.amax(WR_labels) + 1
            select_skeleton = np.where(skel,
                                       WR_labels,
                                       np.ones_like(skel) * superlabel,
                                      )
            skeleton_labels = np.unique(select_skeleton)
            skeleton_bincount = np.bincount(select_skeleton.reshape(-1))
            skeleton_label = np.argmax(skeleton_bincount[:-1])
            WR_labels = np.where(WR_labels != skeleton_label,
                                 WR_labels,
                                 np.zeros_like(WR_labels),
                                )
            WR_labels = label(WR_labels)
        if debug_output:
            print("first round WR_labels debug")
            self.image_stash['debug_display'] = g_img
            self.feature_stash['debug_basins'] = WR_labels
            self.display(tag_in='debug_display',
                         basins_feature='debug_basins',
                         figsize=10,
                         display_labels=True,
                        )
            WR_unique = tuple(np.unique(WR_labels))
            smallest_WR_label = min(WR_unique)
            largest_WR_label = max(WR_unique)
            print("WR_labels: " + str(smallest_WR_label) + " through "
                  + str(largest_WR_label))
        if debug_output:
            pixel_values = R.flatten().tolist()
            plot_target = pixel_values
            obn = 1000
            print("obn = " + str(obn))
            hist, bins = np.histogram(a=plot_target, bins=obn)
            traces = [graph_objs.Scatter(x=bins, y=hist)]
            layout = graph_objs.Layout(plot_bgcolor='rgba(0,0,0,0)',
                                       paper_bgcolor='rgba(0,0,0,0)',
                                       yaxis=dict(title='Count'),
                                       xaxis=dict(title='Pixel value'))
            fig = graph_objs.Figure(data=traces, layout=layout)
            iplot(fig)
        if basin_open_close_size is not None:
            (most_frequent_label,
             background_pixel_coordinates,
             background_pixel_values,
            ) = Plate.most_frequent_label(basins=WR_labels)
            open_closed_basins = Plate.open_close_label_basins(
                                         basins=WR_labels,
                                         open_close_size=basin_open_close_size,
                                         exclude_label=most_frequent_label,
                                                              )
            WR_labels = label(open_closed_basins)
        if debug_output:
            print("openclosed WR_labels debug")
            self.image_stash['debug_display'] = g_img
            self.feature_stash['debug_basins'] = WR_labels
            self.display(tag_in='debug_display',
                         basins_feature='debug_basins',
                         figsize=10,
                         display_labels=True,
                        )
        self.feature_stash[feature_out] = WR_labels
        return None, self.feature_stash[feature_out]

    @staticmethod
    def overlay_labels(waterfall_labels,
                       watershed_labels,
                       debug_output=False,
                      ):
        if waterfall_labels.shape != watershed_labels.shape:
            raise ValueError((waterfall_labels.shape, watershed_labels.shape))
        label_mapper = {}
        label_counter = 1
        overlaid_labels = np.zeros_like(waterfall_labels)
        for (h, w), waterfall_L in np.ndenumerate(waterfall_labels):
            if waterfall_L == 0:
                continue
            watershed_L = watershed_labels[h, w]
            if watershed_L == 0:
                continue
            if (waterfall_L, watershed_L) not in label_mapper:
                label_mapper[(waterfall_L, watershed_L)] = label_counter
                label_counter += 1
            overlaid_labels[h, w] = label_mapper[(waterfall_L, watershed_L)]
        return overlaid_labels

    def overlay_watershed(self,
                          tag_in,
                          intensity_image_tag='intensity_image',
                          median_radius=None,
                          filter_basins=False,
                          waterfall_basins_feature='waterfall_basins',
                          feature_out='overlaid_watershed_basins',
                          min_localmax_dist=10,
                          smoothing_sigma=0,
                          min_area=10,
                          min_intensity=1,
                          rp_radius_factor=0.5,
                          basin_open_close_size=10,
                          debug_output=False,
                          multiplier=1,
                         ):
        g_img = rgb2gray(self.image_stash[tag_in])
        if smoothing_sigma > 0:
            g_img = gaussian(g_img, sigma=smoothing_sigma)
        ng_img = np.amax(g_img) - g_img
        local_maxi = peak_local_max(ng_img, indices=False,
                                    min_distance=min_localmax_dist)
        markers = label(local_maxi)
        WS_labels = watershed(g_img, markers=markers)
        if debug_output:
            print("watershed labels debug")
            self.image_stash['debug_display'] = g_img
            self.feature_stash['debug_basins'] = WS_labels
            self.display(tag_in='debug_display',
                         basins_feature='debug_basins',
                         figsize=10,
                         display_labels=True,
                        )
        WR_labels = self.feature_stash[waterfall_basins_feature]
        overlaid_labels = Plate.overlay_labels(waterfall_labels=WR_labels,
                                               watershed_labels=WS_labels,
                                               debug_output=debug_output,
                                              )
        if debug_output:
            print("overlaid labels debug")
            self.image_stash['debug_display'] = g_img
            self.feature_stash['debug_basins'] = overlaid_labels
            self.display(tag_in='debug_display',
                         basins_feature='debug_basins',
                         figsize=10,
                         display_labels=True,
                        )
        intensity_image = rgb2gray(self.image_stash[intensity_image_tag])
        RP = regionprops(overlaid_labels, intensity_image=intensity_image)
        if median_radius is not None:
            median_intensity_image = median(intensity_image,
                                            selem=disk(median_radius),
                                           )
        else:
            median_intensity_image = intensity_image
        delete_labels = set()
        #image_height, image_width = g_img.shape
        if filter_basins:
            background_basins = overlaid_labels
        else:
            background_basins = None
        for rp in RP:
            if min_area is not None and rp.area < min_area:
                delete_labels.add(rp.label)
            if min_intensity is not None:
                intensity = Plate.rp_intensity(
                                          rp=rp,
                                          background=median_intensity_image,
                                          background_basins=background_basins,
                                          radius=None,
                                          radius_factor=rp_radius_factor,
                                          #radius=max(image_height,
                                          #           image_width,
                                          #          ),
                                          negative=True,
                                          multiplier=multiplier,
                                              )
                if intensity < min_intensity:
                    if debug_output:
                        print("intensity = " + str(intensity))
                    delete_labels.add(rp.label)
        O, Z = Plate.make_bT_bF(image=overlaid_labels, dtype=np.int)
        for L in list(delete_labels):
            mask = np.where(overlaid_labels == L, Z, O)
            overlaid_labels = np.multiply(overlaid_labels, mask)
        overlaid_labels = overlaid_labels.astype(np.int)
        if debug_output:
            print("filtered overlaid labels debug")
            self.image_stash['debug_display'] = g_img
            self.feature_stash['debug_basins'] = overlaid_labels
            self.display(tag_in='debug_display',
                         basins_feature='debug_basins',
                         figsize=10,
                         display_labels=True,
                        )
        if basin_open_close_size is not None:
            (most_frequent_label,
             background_pixel_coordinates,
             background_pixel_values,
            ) = Plate.most_frequent_label(basins=overlaid_labels)
            open_closed_basins = Plate.open_close_label_basins(
                                         basins=overlaid_labels,
                                         open_close_size=basin_open_close_size,
                                         exclude_label=most_frequent_label,
                                                              )
            overlaid_labels = label(open_closed_basins)
            if debug_output:
                print("openclosed overlaid labels debug")
                self.image_stash['debug_display'] = g_img
                self.feature_stash['debug_basins'] = overlaid_labels
                self.display(tag_in='debug_display',
                             basins_feature='debug_basins',
                             figsize=10,
                             display_labels=True,
                            )
        self.feature_stash[feature_out] = overlaid_labels
        return None, self.feature_stash[feature_out]

    def find_lines(self,
                   tag_in,
                   feature_out='lines',
                   hough_threshold=10,
                   smoothing_sigma=1,
                   line_length_factor=0.60,
                   line_gap_factor=0.1,
                  ):
        image = self.image_stash[tag_in]
        g_img = rgb2gray(image)
        height, width = g_img.shape
        if smoothing_sigma > 0:
            gg_img = gaussian(g_img, sigma=smoothing_sigma)
        else:
            gg_img = g_img.copy()
        sg_img = sobel(gg_img)
        tsg_img = sg_img > threshold_otsu(sg_img)
        line_length = width * line_length_factor
        line_gap = line_length * line_gap_factor
        lines = probabilistic_hough_line(tsg_img,
                                         threshold=hough_threshold,
                                         line_length=line_length,
                                         line_gap=line_gap,
                                        )
        self.feature_stash[feature_out] = lines
        return None, self.feature_stash[feature_out]

    def extend_lines(self,
                     tag_in,
                     lines_feature='lines',
                     feature_out='extended_lines',
                    ):
        image = self.image_stash[tag_in]
        lines = self.feature_stash[lines_feature]
        extended_lines = [Plate.extend_line(line=line, image=image)
                          for line in lines]
        self.feature_stash[feature_out] = tuple(extended_lines)
        return None, self.feature_stash[feature_out]

    @staticmethod
    def line_segments_angle(segment_A, segment_B):
        (xA1, yA1), (xA2, yA2) = segment_A
        (xB1, yB1), (xB2, yB2) = segment_B
        vector_A = xA2 - xA1, yA2 - yA1
        vector_B = xB2 - xB1, yB2 - yB1
        AdotB = np.dot(vector_A, vector_B)
        Amag, Bmag = np.linalg.norm(vector_A), np.linalg.norm(vector_B)
        cos_angle = AdotB / (Amag * Bmag)
        angle = degrees(acos(cos_angle))
        #acute_angle = min(angle % 180, 180 - angle % 180)
        #return acute_angle
        return angle

    @staticmethod
    def standard_line_angle(line):
        (w1, h1), (w2, h2) = (x1, y1), (x2, y2) = line
        if h2 >= h1:
            standard_line_segment = ((h1, w1), (h2, w2))
        else:
            standard_line_segment = ((h2, w2), (h1, w1))
        standard_image_segment = ((0, 0), (0, 1))
        angle = Plate.line_segments_angle(segment_A=standard_line_segment,
                                          segment_B=standard_image_segment,
                                         )
        return angle

    @staticmethod
    def line_segments_intersect(segment_A,
                                segment_B,
                                error_tolerance=10**-5,
                               ):
        (xA1, yA1), (xA2, yA2) = segment_A
        (xB1, yB1), (xB2, yB2) = segment_B
        xA, yA = xA2 - xA1, yA2 - yA1
        xB, yB = xB2 - xB1, yB2 - yB1
        denominator = yB * xA - xB * yA
        if denominator == 0:
            #parallel segments
            line_distance = Plate.point_line_distance(point=(xA1, yA1),
                                                      line=segment_B)
            if line_distance < error_tolerance:
                intersect = True
            else:
                intersect = False
        else:
            numerator_A = xB * (yA1 - yB1) - yB * (xA1 - xB1)
            numerator_B = xA * (yA1 - yB1) - yA * (xA1 - xB1)
            uA, uB = numerator_A / denominator, numerator_B / denominator
            if 0 <= uA <= 1 and 0 <= uB <= 1:
                intersect = True
            else:
                intersect = False
        return intersect

    def bundle_lines(self,
                     tag_in,
                     lines_feature='extended_lines',
                     feature_out='bundled_lines',
                     merge_dilation=3,
                     angle_tolerance=10,
                     debug_display=False,
                    ):
        image = rgb2gray(self.image_stash[tag_in])
        image_height, image_width = image.shape
        lines = self.feature_stash[lines_feature]
        line_angles = {line: Plate.standard_line_angle(line=line)
                       for line in lines}
        median_line_angle = np.median(line_angles.values())
        filtered_lines = [line for line, angle in line_angles.iteritems()
                          if abs(angle - median_line_angle) <= angle_tolerance]
        lines_boolean = np.zeros_like(image, dtype=np.bool)
        for line in filtered_lines:
            (w1, h1), (w2, h2) = line
            h1, w1 = int(round(h1)), int(round(w1))
            h2, w2 = int(round(h2)), int(round(w2))
            h1 = min(max(h1, 0), image_height - 1)
            w1 = min(max(w1, 0), image_width - 1)
            h2 = min(max(h2, 0), image_height - 1)
            w2 = min(max(w2, 0), image_width - 1)
            rr, cc = draw.line(h1, w1, h2, w2)
            lines_boolean[rr, cc] = True
        lines_boolean = binary_dilation(image=lines_boolean,
                                        selem=disk(merge_dilation),
                                       )
        lines_boolean = skeletonize(lines_boolean)
        lines_labels = label(lines_boolean)
        if debug_display:
            self.image_stash['debug_display'] = \
                                      np.ones_like(image, dtype=np.uint8) * 250
            self.feature_stash['debug_basins'] = lines_labels
            self.display(tag_in='debug_display',
                         basins_feature='debug_basins',
                         figsize=10,
                        )
        coord_dict = defaultdict(list)
        for (h, w), L in np.ndenumerate(lines_labels):
            if L == 0:
                continue
            coord_dict[L].append((h, w))
        distance_dict = {}
        for L, coords in coord_dict.iteritems():
            distances = Plate.all_pairwise_distances(points=coords)
            (p1, p2), largest_distance = max(distances.items(),
                                             key=lambda x:x[1])
            p1, p2 = p1[::-1], p2[::-1]
            distance_dict[L] = (p1, p2, largest_distance)
        bundled_lines = [Plate.extend_line(line=(p1, p2), image=image)
                         for L, (p1, p2, largest_distance)
                         in distance_dict.iteritems()]
        self.feature_stash[feature_out] = tuple(bundled_lines)
        return None, self.feature_stash[feature_out]

    @staticmethod
    def point_line_distance(point, line):
        """
        point: (x, y)
        line: ((x1, y1), (x2, y2))

        distance = ||(a - p) - ((a - p).(a - b))(a - b) / ||a - b||^2 ||,
        where p = (x, y), a = (x1, y1), b = (x2, y2), and ||X|| is the
        Euclidean norm
        """
        p = point
        a, b = line
        p, a, b = np.array(p), np.array(a), np.array(b)
        u = (a - b) / np.linalg.norm(a - b)
        numerator_vector = (a - p) - np.dot((a - p), u) * u
        numerator_norm = np.linalg.norm(numerator_vector)
        return float(numerator_norm)

    def smooth_lines(self,
                     tag_in,
                     tag_out='smoothed_lines',
                     lines_feature='lines',
                     smooth_radius=3,
                     debug_output=False,
                    ):
        image = self.image_stash[tag_in]
        lines = self.feature_stash[lines_feature]
        g_img = rgb2gray(image)
        lines_boolean = np.zeros_like(g_img, dtype=np.bool)
        for line in lines:
            (w1, h1), (w2, h2) = (x1, y1), (x2, y2) = line
            rr, cc = draw.line(h1, w1, h2, w2)
            lines_boolean[rr, cc] = True
        lines_boolean = binary_dilation(image=lines_boolean,
                                        selem=disk(smooth_radius),
                                       )
        if debug_output:
            print("lines boolean debug")
            self.image_stash['debug_display'] = lines_boolean
            self.display(tag_in='debug_display',
                         figsize=10,
                        )
        smoothing_template = median_filter(g_img, size=smooth_radius)
        smoothed_image = np.where(lines_boolean,
                                  smoothing_template,
                                  g_img,
                                 )
        self.image_stash[tag_out] = smoothed_image
        return self.image_stash[tag_out], None

    @staticmethod
    def points_colinear(points, error_tolerance=10**-5):
        points = sorted(points, key=lambda x:x[0])
        h_coordinates, w_coordinates = zip(*points)
        #Special cases: two ponits, vertical or horizontal line
        if len(points) == 2:
            colinear = True
        elif len(set(h_coordinates)) == 1 or len(set(w_coordinates)) == 1:
            colinear = True
        else:
            #This works because points sorted above and confirmed
            #not to be vertical or horizontal
            (a_h, a_w), (b_h, b_w) = points[0], points[-1]
            #Vertical slope special case taken care of above
            slope = float(b_h - a_h) / (b_w - a_w)
            expected_h_coordinates = [a_h + slope * (w - a_w)
                                      for (h, w) in points]
            errors = [abs(expected_h - h_coordinates[i])
                      for i, expected_h in enumerate(expected_h_coordinates)]
            if max(errors) > error_tolerance:
                colinear = False
            else:
                colinear = True
        return colinear

    @staticmethod
    def all_pairwise_distances(points):
        distances = {(p1, p2): np.linalg.norm(np.array(p1) - np.array(p2))
                     for p1, p2 in product(points, repeat=2)}
        return distances

    @staticmethod
    def find_largest_distance(points, method="naive"):
        if method == "naive":
            distances = Plate.all_pairwise_distances(points=points)
            largest_distance = max(distances.values())
        elif method == "convex_hull":
            raise NotImplementedError("For small number of points, 'naive' "
                                      "will do.")
            if Plate.points_colinear(points):
                #Special case: points are on the same line
                points = sorted(points, key=lambda x:np.linalg.norm(x))
                largest_distance = np.linalg.norm(points[-1] - points[0])
            else:
                hull = ConvexHull(points)
                #There is an O(n) algorithm that can use the hull to find
                #the most distant points
        else:
            raise ValueError("Undefined method.")
        return largest_distance

    @staticmethod
    def grid_hough(points):
        """
        points: [(h1, w1), (h2, w2), (h3, w3), ...]

        Returns optimal grid angle in degrees.
        """
        #Make all unit vectors representing all possible grid angles
        thetas = np.deg2rad(np.arange(0, 180))
        sin_cache, cos_cache = np.sin(thetas), np.cos(thetas)
        unit_vectors = zip(cos_cache, sin_cache)
        perpendicular_unit_vectors = [(unit_vectors[i], unit_vectors[i + 90])
                                      for i in range(90)]
        #Calculate largest distance between two points
        largest_distance = Plate.find_largest_distance(points=points,
                                                       method='naive')
        #Compute total distance metric for all grid angles
        distance_metrics = {}
        for g, grid_archetype in enumerate(perpendicular_unit_vectors):
            point_grid_archetypes = []
            for point in points:
                h, w = point
                point_grid_archetype = tuple([((h, w),
                                               (h + unit_vector_h,
                                                w + unit_vector_w))
                                              for unit_vector_h, unit_vector_w
                                              in grid_archetype])
                point_grid_archetypes.append(point_grid_archetype)
            total_distances = 0
            for p, point in enumerate(points):
                minimal_distance_to_grid = largest_distance
                for p2, grid_archetype in enumerate(point_grid_archetypes):
                    if p == p2:
                        #Avoid comparing point to its own grid archetype
                        continue
                    u1, u2 = grid_archetype
                    u1_distance = Plate.point_line_distance(point=point,
                                                            line=u1)
                    u2_distance = Plate.point_line_distance(point=point,
                                                            line=u2)
                    min_u_distance = min(u1_distance, u2_distance)
                    minimal_distance_to_grid = min(minimal_distance_to_grid,
                                                   min_u_distance)
                total_distances += minimal_distance_to_grid
            distance_metrics[g] = total_distances
        optimal_angle = min(distance_metrics, key=distance_metrics.get)
        return optimal_angle

    @staticmethod
    def generate_rotation_matrix(angle):
        """angle in degrees"""
        angle_radians = radians(angle)
        s, c = np.sin(angle_radians), np.cos(angle_radians)
        rotation_matrix = np.array([[c, -s],
                                    [s,  c]])
        return rotation_matrix

    @staticmethod
    def rotate_points(points, angle):
        """angle in degrees"""
        rotation_matrix = Plate.generate_rotation_matrix(angle=angle)
        rotated_points = [tuple(np.dot(rotation_matrix, np.array(point)))
                          for point in points]
        return tuple(rotated_points)

    def remove_background_basins(self,
                                 tag_in,
                                 basins_feature='waterfall_basins',
                                 feature_out='strongest_basins',
                                 z=2,
                                 open_close_size=10,
                                 debug_output=False,
                                ):
        g_img = rgb2gray(self.image_stash[tag_in])
        basins = self.feature_stash[basins_feature]
        if g_img.shape != basins.shape:
            raise ValueError((g_img.shape, basins.shape))
        (most_frequent_label,
         background_pixel_coordinates,
         background_pixel_values,
        ) = Plate.most_frequent_label(basins=basins, image=g_img)
        background_mu = np.mean(background_pixel_values)
        background_sigma = np.std(background_pixel_values)
        background_threshold = background_mu - background_sigma * z
        bT, bF = Plate.make_bT_bF(image=g_img)
        foreground_bool = np.where(basins != most_frequent_label, bT, bF)
        if debug_output:
            print("foreground_bool debug")
            self.image_stash['debug_display'] = g_img
            self.feature_stash['debug_basins'] = foreground_bool
            self.display(tag_in='debug_display',
                         basins_feature='debug_basins',
                         figsize=10,
                         display_labels=True,
                        )
        remove_bool = np.where(g_img > background_threshold, bT, bF)
        remaining_bool = foreground_bool * ~remove_bool
        closed_opened_bool = Plate.open_close_boolean_basins(
                                               boolean_basins=remaining_bool,
                                               open_close_size=open_close_size,
                                                            )
        strongest_basins = label(closed_opened_bool)
        self.feature_stash[feature_out] = strongest_basins
        return None, self.feature_stash[feature_out]

    @staticmethod
    def bounding_hypercube(points):
        X = zip(*points)
        minmax_pairs = tuple([(min(coordinates), max(coordinates))
                              for coordinates in X])
        return minmax_pairs

    @staticmethod
    def Wk(clustered_points):
        sums_of_pairwise_distances = {cluster: sum(pdist(points))
                                      for cluster, points
                                      in clustered_points.iteritems()}
        cluster_sizes = {cluster: len(points)
                         for cluster, points in clustered_points.iteritems()}
        return sum([Dr / (2.0 * cluster_sizes[cluster])
                    for cluster, Dr in sums_of_pairwise_distances.iteritems()])

    @staticmethod
    def fit_predict_dict(kmeans_assignments,
                         points,
                        ):
        clustered_points = defaultdict(list)
        for cluster, point in zip(kmeans_assignments, points):
            clustered_points[cluster].append(point)
        return clustered_points

    @staticmethod
    def gap_statistic(clustered_points,
                      num_ref_datasets=10,
                     ):
        """
        clustered_points: {cluster_id: (point1, point2, ..., point_i)}

        ... where all points and cluster centers are represented as coordinate
        tuples.


        Gap statistic from

        Tibshirani, Robert, Guenther Walther, and Trevor Hastie. "Estimating
        the number of clusters in a data set via the gap statistic." Journal of
        the Royal Statistical Society: Series B (Statistical Methodology) 63.2
        (2001): 411-423. DOI: 10.1111/1467-9868.00293
        """
        data_Wk = Plate.Wk(clustered_points=clustered_points)
        all_points = sum(clustered_points.values(), [])
        minmax_pairs = Plate.bounding_hypercube(all_points)
        num_points = sum([len(points)
                          for cluster, points
                          in clustered_points.iteritems()])
        ref_kmeans = KMeans(n_clusters=len(clustered_points))
        ref_log_Wks = []
        for d in range(num_ref_datasets):
            random_points = np.array([[uniform(L, U) for L, U in minmax_pairs]
                                      for r in range(num_points)])
            ref_assignments = ref_kmeans.fit_predict(random_points)
            ref_clustered_points = \
                     Plate.fit_predict_dict(kmeans_assignments=ref_assignments,
                                            points=random_points)
            ref_Wk = Plate.Wk(clustered_points=ref_clustered_points)
            ref_log_Wks.append(log(ref_Wk))
        ref_log_Wks_mean = np.mean(ref_log_Wks)
        ref_log_Wks_std = np.std(ref_log_Wks)
        gap_statistic = ref_log_Wks_mean - log(data_Wk)
        sk = sqrt(1 + 1.0 / num_ref_datasets) * ref_log_Wks_std
        return gap_statistic, sk

    @staticmethod
    def PhamDimovNguyen(clustered_points,
                        cluster_centers,
                        prior_S_k,
                        prior_a_k,
                       ):
        """
        clustered_points: {cluster_id: (point1, point2, ..., point_i)}
        cluter_centers: {cluster_id: cluster_center}


        Metric taken from

        Pham, Duc Truong, Stefan S. Dimov, and Chi D. Nguyen. "Selection of K
        in K-means clustering." Proceedings of the Institution of Mechanical
        Engineers, Part C: Journal of Mechanical Engineering Science 219.1
        (2005): 103-119. DOI: 10.1243/095440605X8298
        """
        num_dimensions = len(next(clustered_points.itervalues())[0])
        if num_dimensions < 2:
            raise ValueError("Metric not defined in spaces with less than 2 "
                             "dimensions.")
        distortions = {cluster: sum([euclidean(center, point)**2
                                     for point in clustered_points[cluster]])
                       for cluster, center in cluster_centers.iteritems()}
        S_k = sum(distortions.values())
        k = len(clustered_points)
        if k == 1:
            a_k = None
        elif k == 2:
            a_k = 1 - 3.0 / (4.0 * num_dimensions)
        elif k > 2:
            a_k = prior_a_k + (1 - prior_a_k) / 6.0
        if k == 1:
            f_k = 1
        elif prior_S_k == 0:
            f_k = 1
        elif prior_S_k != 0:
            f_k = S_k / float(a_k * prior_S_k)
        return f_k, S_k, a_k

    @staticmethod
    def determine_k(points,
                    max_k=None,
                    method='jenks',
                    **kwargs
                   ):
        if max_k is None:
            max_k = len(points)
        if method == 'jenks':
            #TODO: 1-cluster case
            gvf_threshold = kwargs.get('gvf_threshold', 0.9)
            gvfs = []
            all_points = sum(points.tolist(), [])
            all_points_mean = np.mean(all_points)
            for n_clusters in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=n_clusters)
                assignments = kmeans.fit_predict(points)
                clustered_points = \
                         Plate.fit_predict_dict(kmeans_assignments=assignments,
                                                points=points)
                cluster_centers = \
                              dict(enumerate(kmeans.cluster_centers_.tolist()))
                #sum of squared deviations from cluster means
                sdcm = np.sum([euclidean(pts, cluster_centers[cluster])**2
                               for cluster, pts
                               in clustered_points.iteritems()
                               for pt in pts])
                #sum of squared deviations from data mean
                sdam = np.sum([euclidean(pt, all_points_mean)**2
                               for cluster, pts
                               in clustered_points.iteritems()
                               for pt in pts])
                gvf = (sdam - sdcm) / sdam
                gvfs.append(gvf)
                if gvf > gvf_threshold:
                    break
            optimal_k = len(gvfs)
        elif method == 'gap':
            num_ref_datasets = kwargs.get('num_ref_datasets', 100)
            gapstats = []
            for n_clusters in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=n_clusters)
                assignments = kmeans.fit_predict(points)
                clustered_points = \
                         Plate.fit_predict_dict(kmeans_assignments=assignments,
                                                points=points)
                gap_stat, sk = Plate.gap_statistic(
                                             clustered_points=clustered_points,
                                             num_ref_datasets=num_ref_datasets,
                                                  )
                if not gapstats:
                    gapstats.append((gap_stat, sk))
                    continue
                prior_gap_stat, prior_sk = gapstats[-1]
                if prior_gap_stat < gap_stat - sk:
                    gapstats.append((gap_stat, sk))
                    continue
                else:
                    break
            optimal_k = len(gapstats)
        elif method == 'PDN':
            metrics = []
            for n_clusters in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=n_clusters)
                assignments = kmeans.fit_predict(points)
                clustered_points = \
                         Plate.fit_predict_dict(kmeans_assignments=assignments,
                                                points=points)
                cluster_centers = \
                              dict(enumerate(kmeans.cluster_centers_.tolist()))
                if not metrics:
                    prior_S_k, prior_a_k = None, None
                else:
                    prior_f_k, prior_S_k, prior_a_k = metrics[-1]
                f_k, S_k, a_k = Plate.PhamDimovNguyen(
                                             clustered_points=clustered_points,
                                             cluster_centers=cluster_centers,
                                             prior_S_k=prior_S_k,
                                             prior_a_k=prior_a_k,
                                                     )
                metrics.append((f_k, S_k, a_k))
            f_ks = [f_k for f_k, S_k, a_k in metrics]
            optimal_k = np.argmin(f_ks)
        else:
            raise ValueError("Undefined method.")
        return optimal_k

    @staticmethod
    def map_sort(items,
                 reverse=False,
                 inverse=False,
                ):
        enumerated_items = tuple(enumerate(items))
        sorted_enumeration = sorted(enumerated_items,
                                    key=lambda x:x[1],
                                    reverse=reverse,
                                   )
        sort_mapping = {original_index: sorted_index
                        for sorted_index, (original_index, item)
                        in enumerate(sorted_enumeration)}
        if inverse:
            sort_mapping = {v: k for k, v in sort_mapping.iteritems()}
        return sort_mapping

    def assign_grid(self,
                    centroids_feature='basin_centroids',
                    feature_out='grid_assignments',
                    **kwargs
                   ):
        centroids = self.feature_stash[centroids_feature]
        centroid_indexes, centroid_coordinates = zip(*centroids.items())
        optimal_grid_angle = Plate.grid_hough(points=centroid_coordinates)
        rotated_centroids = Plate.rotate_points(points=centroid_coordinates,
                                                angle=-optimal_grid_angle)
        rotated_centroids_h, rotated_centroids_w = zip(*rotated_centroids)
        rotated_h_index = dict(zip(centroid_indexes, rotated_centroids_h))
        rotated_w_index = dict(zip(centroid_indexes, rotated_centroids_w))
        rotated_centroids_h = np.array(rotated_centroids_h).reshape(-1, 1)
        rotated_centroids_w = np.array(rotated_centroids_w).reshape(-1, 1)
        h_axis, w_axis = 0, 1
        h_num_k = Plate.determine_k(points=rotated_centroids_h,
                                    max_k=None,
                                    method='jenks',
                                    **kwargs
                                   )
        w_num_k = Plate.determine_k(points=rotated_centroids_w,
                                    max_k=None,
                                    method='jenks',
                                    **kwargs
                                   )
        long_axis = h_axis if h_num_k > w_num_k else w_axis
        num_lanes = min(h_num_k, w_num_k)
        km_lanes = KMeans(n_clusters=num_lanes)
        if long_axis == h_axis:
            to_fit_centroids = rotated_centroids_w
        else:
            to_fit_centroids = rotated_centroids_h
        lane_list = km_lanes.fit_predict(to_fit_centroids)
        lane_coordinates = tuple(km_lanes.cluster_centers_.tolist())
        lane_sort_mapping = Plate.map_sort(items=lane_coordinates,
                                           reverse=False)
        sorted_lane_list = tuple([lane_sort_mapping[lane]
                                  for lane in lane_list])
        lane_assignments = \
                    Plate.fit_predict_dict(kmeans_assignments=sorted_lane_list,
                                           points=centroid_indexes)
        grid_assignments = {}
        for lane, lane_centroid_indexes in lane_assignments.iteritems():
            if long_axis == h_axis:
                lane_centroid_coordinates = [rotated_h_index[index]
                                            for index in lane_centroid_indexes]
            else:
                lane_centroid_coordinates = [rotated_w_index[index]
                                            for index in lane_centroid_indexes]
            sort_mapping = Plate.map_sort(items=lane_centroid_coordinates,
                                          reverse=False,
                                          inverse=True,
                                         )
            sorted_indexes = tuple([lane_centroid_indexes[sort_mapping[i]]
                                    for i, index
                                    in enumerate(lane_centroid_indexes)])
            grid_assignments[lane] = sorted_indexes
        self.feature_stash[feature_out] = grid_assignments
        return None, self.feature_stash[feature_out]

    @staticmethod
    def XYZ2xyY(image):
        X, Y, Z = image[..., 0], image[..., 1], image[..., 2]
        norm = X + Y + Z
        norm = np.where(norm == 0, 1, norm)
        x, y = X / norm, Y / norm
        return np.dstack((x, y, Y))

    def basin_colors(self,
                     tag_in,
                     basins_feature='basins',
                     feature_out='basin_colors',
                     color_space='lab',
                    ):
        rgb_image = self.image_stash[tag_in]
        if color_space == 'rgb':
            color_image = rgb_image
        elif color_space == 'lab':
            color_image = rgb2lab(rgb_image)
        elif color_space == 'hsv':
            color_image = rgb2hsv(rgb_image)
        elif color_space == 'XYZ':
            color_image = rgb2xyz(rgb_image)
        elif color_space == 'xyY':
            color_image = Plate.XYZ2xyY(rgb2xyz(rgb_image))
        elif color_space == 'luv':
            color_image = rgb2luv(rgb_image)
        else:
            raise ValueError("Invalid color space.")
        basins = self.feature_stash[basins_feature]
        basin_pixels = defaultdict(list)
        for (h, w), Label in np.ndenumerate(basins):
            pixel_color = tuple(color_image[h, w].tolist())
            basin_pixels[Label].append(pixel_color)
        basin_pixels = {basin: tuple(pixels)
                        for basin, pixels in basin_pixels.iteritems()}
        self.feature_stash[feature_out] = (color_space, basin_pixels)
        return None, self.feature_stash[feature_out]

    def display_colors_3D(self,
                          basin_colors_feature='basin_colors',
                          subsample=20,
                          height=1000,
                         ):
        color_space, basin_colors = self.feature_stash[basin_colors_feature]
        subsampled_basin_colors = {basin:
                                   sample(pixels, min(subsample, len(pixels)))
                                   for basin, pixels
                                   in basin_colors.iteritems()}
        xyzs = {basin: zip(*pixels)
                for basin, pixels
                in subsampled_basin_colors.iteritems()}
        traces = [graph_objs.Scatter3d(x=x, y=y, z=z,
                                       mode='markers',
                                       marker=dict(size=5),
                                       name = str(basin),
                                      )
                  for basin, (x, y, z) in xyzs.iteritems()]
        if color_space == 'rgb':
            x_title, y_title, z_title = 'R', 'G', 'B'
        elif color_space == 'lab':
            x_title, y_title, z_title = 'L', 'a', 'b'
        elif color_space == 'hsv':
            x_title, y_title, z_title = 'H', 'S', 'V'
        elif color_space == 'XYZ':
            x_title, y_title, z_title = 'X', 'Y', 'Z'
        elif color_space == 'xyY':
            x_title, y_title, z_title = 'x', 'y', 'Y'
        elif color_space == 'luv':
            x_title, y_title, z_title = 'L', 'u', 'v'
        else:
            raise ValueError("Invalid color space.")
        scene=graph_objs.Scene(xaxis=graph_objs.XAxis(title=x_title),
                               yaxis=graph_objs.YAxis(title=y_title),
                               zaxis=graph_objs.ZAxis(title=z_title),
                              )
        layout = graph_objs.Layout(plot_bgcolor='rgba(0,0,0,0)',
                                   paper_bgcolor='rgba(0,0,0,0)',
                                   scene=scene,
                                   height=height,
                                  )
        fig = graph_objs.Figure(data=traces, layout=layout)
        iplot(fig)

    @staticmethod
    def nn_cluster_distance(cluster_A,
                            cluster_B,
                            normalize=True,
                           ):
        """
        1. For each member of cluster_B, find the closest member of cluster_A.
        2. Compute the distance between them.
        3. Returns sum of all such distances.
        """
        nbrs = (NearestNeighbors(n_neighbors=1, n_jobs=-1)
                .fit(np.array(cluster_A))
               )
        distances, indices = nbrs.kneighbors(np.array(cluster_B))
        if normalize:
            cluster_distance = np.mean(distances)
        else:
            cluster_distance = np.sum(distances)
        return cluster_distance

    def mutual_color_distances(self,
                               basin_colors_feature='basin_colors',
                               feature_out='mutual_color_distances',
                               exclude_basins_set=None,
                               include_basins_set=None,
                               normalize=True,
                               sample_size=None,
                              ):
        color_space, basin_colors = self.feature_stash[basin_colors_feature]
        basin_key_set = set(basin_colors)
        if exclude_basins_set is None:
            exclude_basins_set = set()
        else:
            exclude_basins_set = set(exclude_basins_set)
        basin_key_set -= exclude_basins_set
        if include_basins_set is not None:
            include_basins_set = set(include_basins_set)
            basin_key_set &= include_basins_set
        basin_keys = sorted(tuple(basin_key_set))
        mutual_distances = {}
        for basin_A, basin_B in combinations_with_replacement(basin_keys, 2):
            assert (basin_A, basin_B) not in mutual_distances
            if basin_A == basin_B:
                mutual_distances[(basin_A, basin_B)] = 0.0
            else:
                pixels_A = basin_colors[basin_A]
                pixels_B = basin_colors[basin_B]
                if sample_size is not None:
                    sample_size_A = min(sample_size, len(pixels_A))
                    sample_size_B = min(sample_size, len(pixels_B))
                    pixels_A = sample(pixels_A, sample_size_A)
                    pixels_B = sample(pixels_B, sample_size_B)
                mutual_distance = Plate.nn_cluster_distance(
                                                           pixels_A,
                                                           pixels_B,
                                                           normalize=normalize,
                                                           )
                mutual_distances[(basin_A, basin_B)] = mutual_distance
        self.feature_stash[feature_out] = mutual_distances
        return None, self.feature_stash[feature_out]

    def rescale_image(self,
                      tag_in,
                      tag_out,
                      scaling_factor=None,
                      target_height=None,
                      target_width=None,
                     ):
        if ((scaling_factor is None)
            ^ (target_height is None)
            ^ (target_width is None)
           ):
            raise ValueError("Scaling parameters ambiguous.")
        image = self.image_stash[tag_in]
        image_height, image_width = image.shape[:2]
        if target_height is not None:
            scaling_factor = float(target_height) / image_height
        elif target_width is not None:
            scaling_factor = float(target_width) / image_width
        rescaled_image = rescale(image=image, scale=scaling_factor)
        self.image_stash[tag_out] = rescaled_image
        return self.image_stash[tag_out], None

    @staticmethod
    def is_between(x, y, p1, p2):
        p1x, p1y = p1
        p2x, p2y = p2
        if (p1x <= x <= p2x) and (p1y <= y <= p2y):
            return True
        else:
            return False

    @staticmethod
    def fit_segments(chromaticity, calibration_segments):
        x, y = chromaticity
        euclidean_distances = {(cw1, cw2): (euclidean((x, y), (cx1, cy1)),
                                            euclidean((x, y), (cx2, cy2),
                                           )
                                           )
                               for s, (cw1, cw2, (cx1, cy1), (cx2, cy2))
                               in enumerate(calibration_segments)
                              }
        between = {s: Plate.is_between(x, y, (cx1, cy1), (cx2, cy2))
                   for s, (cw1, cw2, (cx1, cy1), (cx2, cy2))
                   in enumerate(calibration_segments)
                  }
        between_index = [s for s, b in between.iteritems() if b]
        num_is_between = len(between_index)
        if num_is_between == 1:
            between_w1, between_w2 = calibration_segments[between_index[0]][:2]
        else:
            inverse_distance_map = defaultdict(list)
            for (cw1, cw2), (e1, e2) in euclidean_distances.iteritems():
                inverse_distance_map[e1].append(cw1)
                inverse_distance_map[e2].append(cw2)
            smallest_distance = min(inverse_distance_map)
            between_w1 = between_w2 = inverse_distance_map[smallest_distance][0]
        #This is a hack that can be done in a much cleaner fashion.
        calibration_wells = ([cw1
                              for s, (cw1, cw2, (cx1, cy1), (cx2, cy2))
                              in enumerate(calibration_segments)]
                             + [calibration_segments[-1][1]]
                            )
        num_wells = len(calibration_wells)
        num_left = 0
        for well in calibration_wells:
            if well != between_w1:
                num_left += 1
            else:
                break
        num_right = num_wells - num_left
        return between_w1, between_w2, num_left, num_right

    @staticmethod
    def point_line_distance_v2(point, line):
        p1, p2 = line
        return norm(np.cross(p2 - p1, p1 - point))/norm(p2 - p1)

    @staticmethod
    def point_line_segment_distance(point, line_segment):
        pL = Plate.point_line_distance_v2(point=point, line=line_segment)
        p1, p2 = line_segment
        p1d, p2d = euclidean(point, p1), euclidean(point, p2)
        return min(p1d, p2d, pL)

    @staticmethod
    def project_point_on_segment(point,
                                 segment,
                                ):
        point = shapely.geometry.Point(point)
        segment = shapely.geometry.LineString(segment)
        np = segment.interpolate(segment.project(point))
        return np.x, np.y

    @staticmethod
    def fit_segments_v2(chromaticity,
                        calibration_segments,
                       ):
        projections = {(cw1, cw2): Plate.project_point_on_segment(
                                              point=chromaticity,
                                              segment=((cx1, cy1), (cx2, cy2)),
                                                                 )
                       for cw1, cw2, (cx1, cy1), (cx2, cy2)
                       in calibration_segments
                      }
        distances = {(cw1, cw2): euclidean(chromaticity, projected_point)
                     for (cw1, cw2), projected_point in projections.iteritems()
                    }
        best_segment = min(distances, key=distances.get)
        best_projection = projections[best_segment]
        segment_coordinates = {(cw1, cw2): ((cx1, cy1), (cx2, cy2))
                               for cw1, cw2, (cx1, cy1), (cx2, cy2)
                               in calibration_segments
                              }
        left_coords, right_coords = segment_coordinates[best_segment]
        left_distance = euclidean(best_projection, left_coords)
        right_distance = euclidean(best_projection, right_coords)
        return best_projection, best_segment, left_distance, right_distance


def find_blobs(go_image,
               summed_image=None,
               LOG_threshold=0.04,
               LOG_min_sigma=5,
               LOG_max_sigma=50,
               LOG_num_sigma=10,
               LOG_overlap=0.5,
               LOG_log_scale=False,
             ):
    """input images assumed to be grayscale."""
    blobs_log = blob_log(image=go_image,
                         min_sigma=LOG_min_sigma,
                         max_sigma=LOG_max_sigma,
                         num_sigma=LOG_num_sigma,
                         threshold=LOG_threshold,
                         overlap=LOG_overlap,
                         log_scale=LOG_log_scale,
                        )
    if summed_image is not None:
        summed_blobs_log = blob_log(image=summed_image,
                                    min_sigma=LOG_min_sigma,
                                    max_sigma=LOG_max_sigma,
                                    num_sigma=LOG_num_sigma,
                                    threshold=LOG_threshold,
                                    overlap=LOG_overlap,
                                    log_scale=LOG_log_scale,
                                   )
    else:
        summed_blobs_log = []
    return tuple(blobs_log), tuple(summed_blobs_log)


def segment_cell_3D(plates,
                    selem_size=5,
                    collapse_threshold=2,
                    image_tag='go_image',
                   ):
    image_stack = [plate.image_stash[image_tag] for plate in plates]
    segment_stack = [image > threshold_otsu(image) for image in image_stack]
    int_segment_stack = [ti.astype(dtype=np.int) for ti in segment_stack]
    collapsed_stack = sum(int_segment_stack)
    segmented_collapse = collapsed_stack >= collapse_threshold
    labeled_collapse = label(segmented_collapse)
    label_set = set(labeled_collapse.reshape(-1).tolist())
    label_volumes = {Label: np.sum([np.where(labeled_collapse == Label,
                                             collapsed_stack,
                                             0,
                                            )
                                   ]
                                  )
                     for Label in iter(label_set)
                    }
    largest_label = max(label_volumes, key=label_volumes.get)
    segment_template = np.where(labeled_collapse == largest_label, True, False)
    propagated_segments = [segment_template * z_layer for z_layer in segment_stack]      
    labeled_propagations = [label(z_layer) for z_layer in propagated_segments]
    propagation_RPs = [regionprops(label_image=layer,
                                   intensity_image=image_stack[z]
                                  )
                       for z, layer in enumerate(labeled_propagations)
                      ]
    propagation_labels = [max(RP, key=lambda rp:rp.area).label
                          for RP in propagation_RPs
                         ]
    segments = [np.where(Labels == propagation_labels[z],
                         True,
                         False,
                        )
                for z, Labels in enumerate(labeled_propagations)
               ]
    if selem_size is not None:
        selem = disk(selem_size)
        segments = [binary_opening(binary_closing(segment, selem=selem),
                                   selem=selem,
                                  )
                    for segment in segments]
    return segments

def make_husk(plate,
              image_tag='go_image',
              segment_tag='cell_segment',
              median_size=51,
              selem_size=2,
              median_fill=False,
              null_segment=False,
             ):
    selem = disk(selem_size)
    go_image = plate.image_stash[image_tag]
    cell_segment = plate.image_stash[segment_tag]
    if median_fill:
        filler = np.median(go_image[cell_segment])
    else:
        filler = 0
    median_prep = np.where(cell_segment, go_image, filler)
    mf_img = median_filter(median_prep, median_size)
    sm_img = np.maximum(np.subtract(median_prep, mf_img), 0)
    co_img = closing(opening(image=sm_img, selem=selem), selem=selem)
    if null_segment:
        co_img = np.where(cell_segment, co_img, 0)
    return mf_img, sm_img, co_img


def segment_puncta(plate,
                   selem_size=2,
                   median_size=51,
                   markers_closing_radius=None,
                   watershed_compactness=0,
                   mf_img_tag=None,
                   sm_img_tag=None,
                   co_img_tag=None,
                   go_image_tag='go_image',
                   cell_segment_tag='cell_segment',
                   maximum_radius=None,
                   maximum_radius_ratio=None,
                  ):
    selem = disk(selem_size)
    cell_segment = plate.image_stash[cell_segment_tag]
    tag_check = [t is None for t in [mf_img_tag, sm_img_tag, co_img_tag]]
    if any(tag_check) and not all(tag_check):
        raise ValueError("Either all of mf_img_tag, sm_img_tag, co_img_tag "
                         "are None, or none are."
                        )
    if co_img_tag is None:
        go_image = plate.image_stash[go_image_tag]
        median_prep = np.where(cell_segment, go_image, 0)
        mf_img = median_filter(median_prep, median_size)
        sm_img = np.maximum(np.subtract(median_prep, mf_img), 0)
        co_img = closing(opening(image=sm_img, selem=selem), selem=selem)
    else:
        mf_img = plate.image_stash[mf_img_tag]
        sm_img = plate.image_stash[sm_img_tag]
        co_img = plate.image_stash[co_img_tag]
    inverse_co_img = invert(co_img)
    blobs_log = plate.feature_stash['blobs_log']
    local_maxima = np.zeros_like(inverse_co_img, dtype=np.bool)
    for h, w, sigma in blobs_log:
        local_maxima[int(h), int(w)] = True
    summed_blobs_log = plate.feature_stash.get('summed_blobs_log', None)
    if summed_blobs_log is not None:
        for h, w, sigma in summed_blobs_log:
            local_maxima[int(h), int(w)] = True
    if markers_closing_radius is not None:
        selem = disk(markers_closing_radius)
        local_maxima = binary_closing(image=local_maxima,
                                      selem=selem,
                                     )
    markers = label(local_maxima)
    W_labels = watershed(inverse_co_img,
                         markers=markers,
                         compactness=watershed_compactness,
                        )
    W_labels = label(np.where(cell_segment, W_labels, 0))
    W_labels = label(np.where(co_img > 0, W_labels, 0))
    all_labels = set(W_labels.reshape(-1).tolist())
    valid_labels = set([W_labels[int(h), int(w)]
                        for h, w, sigma in blobs_log])
    if summed_blobs_log is not None:
        valid_labels |= set([W_labels[int(h), int(w)]
                             for h, w, sigma in summed_blobs_log])
    invalid_labels = all_labels - valid_labels
    for L in tuple(invalid_labels):
        W_labels = np.where(W_labels == L, 0, W_labels)
    if maximum_radius_ratio is not None:
        for h, w, sigma in blobs_log:
            image_h, image_w = W_labels.shape
            x, y = np.mgrid[0 - h:image_h - h, 0 - w:image_w - w]
            distances = np.sqrt(x**2 + y**2)
            Label = W_labels[int(h), int(w)]
            elimination_mask = \
                            (np.where(W_labels == Label,
                                      True,
                                      False,
                                     )
                           * np.where(distances > maximum_radius_ratio * sigma,
                                      True,
                                      False,
                                     )
                            )
            W_labels = np.where(elimination_mask, 0, W_labels)
    if maximum_radius is not None:
        for h, w, sigma in blobs_log:
            image_h, image_w = W_labels.shape
            x, y = np.mgrid[0 - h:image_h - h, 0 - w:image_w - w]
            distances = np.sqrt(x**2 + y**2)
            Label = W_labels[int(h), int(w)]
            elimination_mask = (np.where(W_labels == Label,
                                         True,
                                         False,
                                        )
                                * np.where(distances > maximum_radius,
                                           True,
                                           False,
                                          )
                               )
            W_labels = np.where(elimination_mask, 0, W_labels)
            
    return W_labels, mf_img, sm_img, co_img


def median_images(go_image,
                  cell_segment,
                  median_size=51,
                 ):
    raise DeprecationWarning("segment_puncta_MP now saves these on the go.")
    median_prep = np.where(cell_segment, go_image, 0)
    mf_img = median_filter(median_prep, median_size)
    sm_img = np.maximum(np.subtract(median_prep, mf_img), 0)
    return mf_img, sm_img


def build_3D_puncta(plates,
                    channel,
                    husk_tag='husk_watershed_basins',
                   ):
    label_graph = nx.Graph()
    for p, (plate_1, plate_2) in enumerate(Plate.pairwise(plates)):
        labels_1 = plate_1.feature_stash[husk_tag]
        labels_2 = plate_2.feature_stash[husk_tag]
        for (h, w), L in np.ndenumerate(labels_1):
            if L != 0:
                label_graph.add_node((p, channel, L))
        for (h, w), L in np.ndenumerate(labels_2):
            if L != 0:
                label_graph.add_node((p + 1, channel, L))
        labels_1_coordinates = defaultdict(set)
        for (h, w), L in np.ndenumerate(labels_1):
            if L != 0:
                labels_1_coordinates[L].add((h, w))
        labels_2_coordinates = defaultdict(set)
        for (h, w), L in np.ndenumerate(labels_2):
            if L != 0:
                labels_2_coordinates[L].add((h, w))
        for L1, coord_set_1 in labels_1_coordinates.iteritems():
            for L2, coord_set_2 in labels_2_coordinates.iteritems():
                if not coord_set_1.isdisjoint(coord_set_2):
                    label_graph.add_edge((p, channel, L1),
                                         (p + 1, channel, L2)
                                        )
    return label_graph


def find_puncta_overlaps(RFP_plates,
                         GFP_plates,
                         husk_tag='husk_watershed_basins',
                         RFP_label_tag='RFP_label_graph',
                         GFP_label_tag='GFP_label_graph',
                        ):
    for z, (RFP_plate, GFP_plate) in enumerate(izip(RFP_plates, GFP_plates)):
        assert (RFP_plate.feature_stash['channel'] == 'RFP'
                and GFP_plate.feature_stash['channel'] == 'GFP'
               )
        RFP_basins = RFP_plate.feature_stash[husk_tag]
        GFP_basins = GFP_plate.feature_stash[husk_tag]
        RFP_label_graph = RFP_plates[0].feature_stash[RFP_label_tag]
        GFP_label_graph = GFP_plates[0].feature_stash[GFP_label_tag]
        R_label_set = set(RFP_basins.reshape(-1).tolist()) - set([0])
        G_label_set = set(GFP_basins.reshape(-1).tolist()) - set([0])
        for R_label in iter(R_label_set):
            overlap_array = np.where(RFP_basins == R_label, 1, 0)
            overlap_array = np.where(GFP_basins != 0, overlap_array, 0)
            GFP_subimage = GFP_basins * overlap_array
            GFP_overlapped_labels = (set(GFP_subimage.reshape(-1).tolist())
                                     - set([0])
                                    )
            overlaps = (z,
                        R_label,
                        overlap_array,
                        GFP_subimage,
                        GFP_overlapped_labels,
                       )
            RFP_puncta = (z, 'RFP', R_label)
            RFP_label_graph.node[RFP_puncta]['overlaps'] = overlaps
        for G_label in iter(G_label_set):
            overlap_array = np.where(GFP_basins == G_label, 1, 0)
            overlap_array = np.where(RFP_basins != 0, overlap_array, 0)
            RFP_subimage = RFP_basins * overlap_array
            RFP_overlapped_labels = (set(RFP_subimage.reshape(-1).tolist())
                                     - set([0])
                                    )
            overlaps = (z,
                        G_label,
                        overlap_array,
                        RFP_subimage,
                        RFP_overlapped_labels,
                       )
            GFP_puncta = (z, 'GFP', G_label)
            GFP_label_graph.node[GFP_puncta]['overlaps'] = overlaps

    RFP_label_graph = RFP_plates[0].feature_stash[RFP_label_tag]
    GFP_label_graph = GFP_plates[0].feature_stash[GFP_label_tag]
    threeD_RFP_overlaps = defaultdict(list)
    threeD_GFP_overlaps = defaultdict(list)
    threeD_RFP_puncta = nx.connected_components(RFP_label_graph)
    threeD_GFP_puncta = nx.connected_components(GFP_label_graph)
    for puncta in threeD_RFP_puncta:
        puncta_key = frozenset(puncta)
        assert puncta_key not in threeD_RFP_overlaps
        for (z, channel, Label) in iter(puncta):
            assert channel == 'RFP'
            overlaps = RFP_label_graph.node[(z, channel, Label)]['overlaps']
            threeD_RFP_overlaps[puncta_key].append(overlaps)
    threeD_RFP_overlaps = {puncta_key: tuple(overlaps_list)
                           for puncta_key, overlaps_list
                           in threeD_RFP_overlaps.iteritems()
                          }
    for puncta in threeD_GFP_puncta:
        puncta_key = frozenset(puncta)
        assert puncta_key not in threeD_GFP_overlaps
        for (z, channel, Label) in iter(puncta):
            assert channel == 'GFP'
            overlaps = GFP_label_graph.node[(z, channel, Label)]['overlaps']
            threeD_GFP_overlaps[puncta_key].append(overlaps)
    threeD_GFP_overlaps = {puncta_key: tuple(overlaps_list)
                           for puncta_key, overlaps_list
                           in threeD_GFP_overlaps.iteritems()
                          }
    return threeD_RFP_overlaps, threeD_GFP_overlaps

def attenuate_edges(plate,
                    edt_limit=20.0,
                    segment_tag='cell_segment',
                    go_tag='go_image',
                   ):
    cell_segment = plate.image_stash[segment_tag]
    cell_segment_edt = ndi.morphology.distance_transform_edt(cell_segment)
    cell_segment_edt[cell_segment_edt > edt_limit] = edt_limit
    cell_segment_edt = cell_segment_edt / float(edt_limit)
    go_image = plate.image_stash[go_tag]
    attenuated_image = cell_segment_edt * go_image
    return attenuated_image


def segment_again(plate,
                  selem_size=5,
                  image_tag='go_image',
                  segment_tag='cell_segment',
                 ):
    go_image = plate.image_stash[image_tag]
    first_segment = plate.image_stash[segment_tag]
    filled_segment = binary_fill_holes(first_segment)
    second_threshold = threshold_otsu(go_image[np.where(filled_segment)])
    second_segment = go_image > second_threshold
    selem = disk(selem_size)
    second_segment = binary_opening(binary_closing(second_segment,
                                                   selem=selem,
                                                  ),
                                    selem=selem,
                                   )
    second_segment *= first_segment
    return second_segment


class PunctaExperiment(object):
    def __init__(self,
                 experiment_directory,
                 load_filter="DS_Store",
                 shape=None,
                ):
        replicate_directories = \
                [os.path.abspath(os.path.join(experiment_directory, replicate))
                 for replicate in os.listdir(experiment_directory)]
        channel_pair_directories = {}
        for replicate in replicate_directories:
            if not os.path.isdir(replicate):
                continue
            channel_pair_directories[replicate] = \
                                  [channel for channel in os.listdir(replicate)
                                   if load_filter not in channel]
        self.replicates = {}
        self.failed_files = defaultdict(list)
        for replicate, (channel_1,
                        channel_2) in channel_pair_directories.iteritems():
            channel_1_dir = os.path.join(replicate, channel_1)
            channel_2_dir = os.path.join(replicate, channel_2)
            channel_1_filenames = \
                  sorted([os.path.abspath(f)
                          for f in glob(os.path.join(channel_1_dir, "*.png"))])
            channel_2_filenames = \
                  sorted([os.path.abspath(f)
                          for f in glob(os.path.join(channel_2_dir, "*.png"))])
            if "GFP" in channel_1 and "GFP" in channel_2:
                raise Exception("Both channels are GFP?")
            elif "GFP" in channel_1:
                GFP_filenames = channel_1_filenames
                RFP_filenames = channel_2_filenames
            elif "GFP" in channel_2:
                GFP_filenames = channel_2_filenames
                RFP_filenames = channel_1_filenames
            else:
                raise Exception("GFP channel not found")
            RFP_plates, GFP_plates = [], []
            for f in RFP_filenames:
                try:
                    image = imread(f)
                    if shape is not None:
                        image = resize(image=image,
                                       output_shape=shape,
                                      )
                    plate = Plate(image=image,
                                  tag_in='original_image',
                                  source_filename=f,
                                 )
                except Exception as e:
                    self.failed_files[replicate].append((f, e))
                else:
                    plate.feature_stash['channel'] = "RFP"
                    RFP_plates.append(plate)
            for f in GFP_filenames:
                try:
                    image = imread(f)
                    if shape is not None:
                        image = resize(image=image,
                                       output_shape=shape,
                                      )
                    plate = Plate(image=image,
                                  tag_in='original_image',
                                  source_filename=f,
                                 )
                except Exception as e:
                    self.failed_files[replicate].append((f, e))
                else:
                    plate.feature_stash['channel'] = "GFP"
                    GFP_plates.append(plate)
            self.replicates[replicate] = (tuple(RFP_plates), tuple(GFP_plates))

    def make_go_images(self):
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            for plate in RFP_plates + GFP_plates:
                plate.image_stash['go_image'] = \
                                  rgb2gray(plate.image_stash['original_image'])

    @staticmethod
    def z_sum(stack,
              depth,
              mean=True,
             ):
        if depth % 2 != 1:
            raise ValueError("depth must be odd")
        d = int((depth - 1) / 2) #int casting necessary due to future division
                                 #to ensure slice indices are integers
        image_cohorts = [(i, stack[i - d:i + d + 1])
                         for i, image in enumerate(stack)
                         if len(stack[i - d:i + d + 1]) == depth]
        if mean:
            cohort_sums = [(i, np.mean(cohort, axis=0))
                           for i, cohort in image_cohorts]
        else:
            cohort_sums = [(i, np.sum(cohort, axis=0))
                           for i, cohort in image_cohorts]
        return tuple(cohort_sums)

    @staticmethod
    def z_sum_stack(z_stack_plates, depth=3):
        z_stack = [plate.image_stash['go_image'] for plate in z_stack_plates]
        z_summed = PunctaExperiment.z_sum(stack=z_stack,
                                          depth=depth,
                                          mean=True,
                                         )
        for i, summed_image in z_summed:
            z_stack_plates[i].image_stash['z_summed'] = summed_image

    def make_z_sums(self,
                    depth=3,
                   ):
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            PunctaExperiment.z_sum_stack(z_stack_plates=RFP_plates,
                                         depth=depth,
                                        )
            PunctaExperiment.z_sum_stack(z_stack_plates=GFP_plates,
                                         depth=depth,
                                        )

    @staticmethod
    def find_blobs(go_image,
                   summed_image=None,
                   LOG_threshold=0.04,
                   LOG_min_sigma=5,
                   LOG_max_sigma=50,
                   LOG_num_sigma=10,
                   LOG_overlap=0.5,
                   LOG_log_scale=False,
                 ):
        """input images assumed to be grayscale."""
        raise DeprecationWarning("Use module-level function.")
        blobs_log = blob_log(image=go_image,
                             min_sigma=LOG_min_sigma,
                             max_sigma=LOG_max_sigma,
                             num_sigma=LOG_num_sigma,
                             threshold=LOG_threshold,
                             overlap=LOG_overlap,
                             log_scale=LOG_log_scale,
                            )
        if summed_image is not None:
            summed_blobs_log = blob_log(image=summed_image,
                                        min_sigma=LOG_min_sigma,
                                        max_sigma=LOG_max_sigma,
                                        num_sigma=LOG_num_sigma,
                                        threshold=LOG_threshold,
                                        overlap=LOG_overlap,
                                        log_scale=LOG_log_scale,
                                       )
        else:
            summed_blobs_log = []
        return tuple(blobs_log), tuple(summed_blobs_log)

    def experiment_blobs(self,
                         GFP_LOG_threshold=0.04,
                         RFP_LOG_threshold=0.02,
                         LOG_min_sigma=5,
                         LOG_max_sigma=50,
                         LOG_num_sigma=10,
                         LOG_overlap=0.5,
                         LOG_log_scale=False,
                         edge_smooth=None,
                         edge_max=False,
                         edge_elimination=None,
                         median_filler=False,
                         segment_elimination=False,
                         go_image_tag='go_image',
                         cell_segment_tag='cell_segment',
                         blobs_log_tag_out='blobs_log',
                         summed_blobs_log_tag_out='summed_blobs_log',
                         omit_replicates=None,
                         num_processes=None,
                        ):
        if edge_max:
            raise DeprecationWarning("This does not work well.")
        pool = multiprocessing.Pool(processes=num_processes,
                                    maxtasksperchild=None,
                                   )
        processes = []
        if omit_replicates is None:
            omit_replicates = set()
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            if replicate in omit_replicates:
                continue
            concatenated_plates = RFP_plates + GFP_plates
            for p, plate in enumerate(concatenated_plates):
                go_image = plate.image_stash[go_image_tag]
                summed_image = plate.image_stash.get('z_summed', None)
                if edge_smooth is not None:
                    cell_segment = plate.image_stash[cell_segment_tag]
                    gaussian_go_image = gaussian(image=go_image,
                                                 sigma=edge_smooth,
                                                )
                    go_image = np.where(cell_segment,
                                        go_image,
                                        gaussian_go_image,
                                       )
                    if summed_image is not None:
                        gaussian_summed_image = gaussian(image=summed_image,
                                                         sigma=edge_smooth,
                                                        )
                        summed_image = np.where(cell_segment,
                                                summed_image,
                                                gaussian_summed_image,
                                               )
                if edge_max:
                    cell_segment = plate.image_stash[cell_segment_tag]
                    max_go_image = np.ones_like(go_image) * np.amax(go_image)
                    go_image = np.where(cell_segment,
                                        go_image,
                                        max_go_image,
                                       )
                    if summed_image is not None:
                        max_summed_image = (np.ones_like(summed_image)
                                            * np.amax(summed_image)
                                           )
                        summed_image = np.where(cell_segment,
                                                summed_image,
                                                max_summed_image,
                                               )
                if median_filler:
                    if edge_max or edge_smooth:
                        raise UserWarning("Using more than one of these "
                                          "options will cause problems.")
                    cell_segment = plate.image_stash[cell_segment_tag]
                    #cell_segment = binary_fill_holes(input=cell_segment)
                    #plate.image_stash['filled_cell_segment'] = cell_segment
                    go_median = np.median(go_image[cell_segment])
                    go_image = go_image.copy()
                    go_image[~cell_segment] = go_median
                    plate.image_stash['filled_go_image'] = go_image.copy()
                    if summed_image is not None:
                        summed_median = np.median(summed_image[cell_segment])
                        summed_image = summed_image.copy()
                        summed_image[~cell_segment] = summed_median
                        plate.image_stash['filled_summed_image'] = \
                                                            summed_image.copy()
                if plate.feature_stash['channel'] == 'RFP':
                    LOG_threshold = RFP_LOG_threshold
                else:
                    LOG_threshold = GFP_LOG_threshold
                process = pool.apply_async(find_blobs,
                                           #PunctaExperiment.find_blobs,
                                           (go_image,
                                            summed_image,
                                            LOG_threshold,
                                            LOG_min_sigma,
                                            LOG_max_sigma,
                                            LOG_num_sigma,
                                            LOG_overlap,
                                            LOG_log_scale,
                                           )
                                          )
                processes.append((replicate, p, process))
        pool.close()
        pool.join()
        for replicate, p, process in processes:
            blobs_log, summed_blobs_log = process.get()
            if edge_elimination is not None:
                cell_segment = plate.image_stash[cell_segment_tag]
                def blob_touches_edge(blob, segment, r_factor=1):
                    h, w, r = blob
                    ih, iw = int(round(h)), int(round(w))
                    H, W = segment.shape
                    y, x = np.ogrid[-ih:H - ih, -iw:W - iw]
                    mask = (x*x + y*y <= r*r * r_factor*r_factor)
                    disk_array = np.zeros_like(segment, dtype=np.bool)
                    disk_array[mask] = True
                    overlap = np.logical_and(~segment, disk_array)
                    if np.any(overlap):
                        touches = True
                    else:
                        touches = False
                    #touches = False
                    #for (H, W), s in np.ndenumerate(segment):
                    #    if s:
                    #        continue
                    #    d = euclidean([h, w], [H, W])
                    #    if d <= r * r_factor:
                    #        touches = True
                    #        break
                    return touches
                blobs_log = [blob
                             for blob in blobs_log
                             if not blob_touches_edge(blob,
                                                      cell_segment,
                                                      edge_elimination,
                                                     )
                            ]
                summed_blobs_log = [blob
                                    for blob in summed_blobs_log
                                    if not blob_touches_edge(blob,
                                                             cell_segment,
                                                             edge_elimination,
                                                            )
                                   ]
            if segment_elimination:
                cell_segment = plate.image_stash[cell_segment_tag]
                ebl = []
                for blob in blobs_log:
                    h, w, r = blob
                    ih, iw = int(round(h)), int(round(w))
                    h_size, w_size = cell_segment.shape
                    if not (0 <= ih < h_size) and (0 <= iw < w_size):
                        continue
                    if cell_segment[ih, iw]:
                        ebl.append(blob)
                blobs_log = ebl
                esbl = []
                for blob in summed_blobs_log:
                    h, w, r = blob
                    ih, iw = int(round(h)), int(round(w))
                    h_size, w_size = cell_segment.shape
                    if not (0 <= ih < h_size) and (0 <= iw < w_size):
                        continue
                    if cell_segment[ih, iw]:
                        esbl.append(blob)
                summed_blobs_log = esbl
            RFP_plates, GFP_plates = self.replicates[replicate]
            concatenated_plates = RFP_plates + GFP_plates
            plate = concatenated_plates[p]
            plate.feature_stash[blobs_log_tag_out] = blobs_log
            plate.feature_stash[summed_blobs_log_tag_out] = summed_blobs_log

    @staticmethod
    def segment_cell(go_image,
                     selem_size=2,
                    ):
        """go_image assumed to be grayscale."""
        selem = disk(selem_size)
        cell_segment = threshold_otsu(go_image) < go_image
        cell_segment = opening(closing(cell_segment, selem=selem), selem=selem)
        labels = label(input=cell_segment)
        RP = regionprops(label_image=labels,
                         intensity_image=go_image,
                        )
        largest_label = max(RP, key=lambda rp:rp.area).label
        cell_segment = np.where(labels == largest_label, True, False)
        return cell_segment

    def segment_all_cells(self,
                          use_channel='RFP',
                          selem_size=2,
                         ):
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            if use_channel == 'RFP':
                primary_plate_stack = RFP_plates
                secondary_plate_stack = GFP_plates
            else:
                primary_plate_stack = GFP_plates
                secondary_plate_stack = RFP_plates
            for p, plate in enumerate(primary_plate_stack):
                go_image = plate.image_stash['go_image']
                cell_segment = PunctaExperiment.segment_cell(
                                                         go_image=go_image,
                                                         selem_size=selem_size,
                                                             )
                plate.image_stash['cell_segment'] = cell_segment
                secondary_plate = secondary_plate_stack[p]
                secondary_plate.image_stash['cell_segment'] = cell_segment

    def segment_puncta_MP(self,
                          selem_size=2,
                          median_size=51,
                          num_processes=None,
                          markers_closing_radius=None,
                          watershed_compactness=0,
                          mf_img_tag=None,
                          sm_img_tag=None,
                          co_img_tag=None,
                          go_image_tag='go_image',
                          cell_segment_tag='cell_segment',
                          maximum_radius=None,
                          maximum_radius_ratio=None,
                         ):
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes,
                                    maxtasksperchild=None,
                                   )
        processes = []
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            concatenated_plates = RFP_plates + GFP_plates
            for p, plate in enumerate(concatenated_plates):
                process = pool.apply_async(segment_puncta,
                                           #PunctaExperiment.segment_puncta,
                                           (plate,
                                            selem_size,
                                            median_size,
                                            markers_closing_radius,
                                            watershed_compactness,
                                            mf_img_tag,
                                            sm_img_tag,
                                            co_img_tag,
                                            go_image_tag,
                                            cell_segment_tag,
                                            maximum_radius,
                                           )
                                          )
                processes.append((replicate, p, process))
        pool.close()
        pool.join()
        for replicate, p, process in processes:
            W_labels, mf_img, sm_img, co_img = process.get()
            RFP_plates, GFP_plates = self.replicates[replicate]
            concatenated_plates = RFP_plates + GFP_plates
            plate = concatenated_plates[p]
            plate.feature_stash['husk_watershed_basins'] = W_labels
            plate.image_stash['mf_img'] = mf_img
            plate.image_stash['sm_img'] = sm_img
            plate.image_stash['co_img'] = co_img

    def make_amplified_images(self,
                              tag_in='go_image',
                              tag_out='amplified_image',
                             ):
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            for plate in RFP_plates + GFP_plates:
                go_image = plate.image_stash[tag_in]
                amplified_image = go_image / np.amax(go_image)
                plate.image_stash[tag_out] = amplified_image

    def generate_median_images_MP(self,
                                  median_size=51,
                                  num_processes=None,
                                 ):
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes,
                                    maxtasksperchild=None,
                                   )
        processes = []
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            concatenated_plates = RFP_plates + GFP_plates
            for p, plate in enumerate(concatenated_plates):
                go_image = plate.image_stash['go_image']
                cell_segment = plate.image_stash['cell_segment']
                process = pool.apply_async(median_images,
                                           (go_image,
                                            cell_segment,
                                            median_size,
                                           )
                                          )
                processes.append((replicate, p, process))
        pool.close()
        pool.join()
        for replicate, p, process in processes:
            mf_img, sm_img = process.get()
            RFP_plates, GFP_plates = self.replicates[replicate]
            concatenated_plates = RFP_plates + GFP_plates
            plate = concatenated_plates[p]
            plate.image_stash['mf_img'] = mf_img
            plate.image_stash['sm_img'] = sm_img

    def inverse_images(self,
                       tag_in='go_image',
                       tag_out='inverse_image',
                      ):
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            for plate in RFP_plates + GFP_plates:
                go_image = plate.image_stash[tag_in]
                inverse_image = np.amax(go_image) - go_image
                plate.image_stash[tag_out] = inverse_image

    def filled_images(self,
                      selem_size_1=2,
                      selem_size_2=5,
                      cell_segment_tag='cell_segment',
                      inverse_image_tag='inverse_image',
                      go_tag='go_image',
                      tag_out='sum_img',
                      ism_tag='ism_img',
                      ico_tag='ico_img',
                     ):
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            for plate in RFP_plates + GFP_plates:
                cell_segment = plate.image_stash[cell_segment_tag]
                filled_segment = binary_fill_holes(cell_segment)
                inverse_image = plate.image_stash[inverse_image_tag]
                imf_img = (np.ones_like(inverse_image)
                           * np.median(inverse_image[filled_segment]))
                ism_img = np.maximum(np.subtract(inverse_image, imf_img), 0)
                selem_1 = disk(selem_size_1)
                ico_img = closing(opening(image=ism_img, selem=selem_1),
                                  selem=selem_1,
                                 )
                selem_2 = disk(selem_size_2)
                ico_img[~binary_dilation(filled_segment, selem=selem_2)] = 0
                plate.image_stash[ism_tag] = ism_img
                plate.image_stash[ico_tag] = ico_img
                go_image = plate.image_stash[go_tag]
                sum_img = go_image + ico_img
                plate.image_stash[tag_out] = sum_img

    def decapitate(self,
                   num_z_layers,
                  ):
        self.replicates = {replicate: (RFP_plates[num_z_layers:],
                                       GFP_plates[num_z_layers:],
                                      )
                           for replicate, (RFP_plates, GFP_plates)
                           in self.replicates.iteritems()
                          }

    def remove_mismatched_replicates(self,
                                     remove_empty_replicates=True,
                                    ):
        self.replicates = {replicate: (RFP_plates, GFP_plates)
                           for replicate, (RFP_plates, GFP_plates)
                           in self.replicates.iteritems()
                           if len(RFP_plates) == len(GFP_plates)
                          }
        if remove_empty_replicates:
            self.replicates = {replicate: (RFP_plates, GFP_plates)
                               for replicate, (RFP_plates, GFP_plates)
                               in self.replicates.iteritems()
                               if len(RFP_plates) > 0
                              }

    def check_resolutions_match(self,
                                tag_in,
                               ):
        image_resolutions = set()
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            for plate in RFP_plates + GFP_plates:
                image = plate.image_stash[tag_in]
                resolution = image.shape
                image_resolutions.add(resolution)
        if len(image_resolutions) == 1:
            match = True
        else:
            match = False
        return match, image_resolutions

    def binary_openclose_segments(self,
                                  selem_radius=5,
                                  segment_tag='cell_segment',
                                 ):
        selem = disk(selem_radius)
        for RFP_plates, GFP_plates in self.replicates.itervalues():
            for plate in RFP_plates + GFP_plates:
                segment = plate.image_stash[segment_tag]
                segment = binary_closing(binary_opening(image=segment,
                                                        selem=selem,
                                                       ),
                                         selem=selem,
                                        )
                plate.image_stash[segment_tag] = segment

    def attenuate_edges(self,
                        edt_limit=20.0,
                        segment_tag='cell_segment',
                        go_tag='go_image',
                        tag_out='edt_img',
                       ):
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            zipped_plates = izip(RFP_plates, GFP_plates)
            for z, (RFP_plate, GFP_plate) in enumerate(zipped_plates):
                for plate in RFP_plate, GFP_plate:
                    cell_segment = plate.image_stash[segment_tag]
                    cell_segment_edt = ndi.morphology.distance_transform_edt(
                                                                   cell_segment
                                                                            )
                    cell_segment_edt[cell_segment_edt > edt_limit] = edt_limit
                    cell_segment_edt = cell_segment_edt / edt_limit
                    go_image = plate.image_stash[go_tag]
                    plate.image_stash[tag_out] = cell_segment_edt * go_image

    def attenuate_edges_MP(self,
                           edt_limit=20.0,
                           segment_tag='cell_segment',
                           go_tag='go_image',
                           tag_out='edt_img',
                           num_processes=None,
                          ):
        pool = multiprocessing.Pool(processes=num_processes,
                                    maxtasksperchild=None,
                                   )
        processes = []
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            for plate in RFP_plates + GFP_plates:
                process = pool.apply_async(attenuate_edges,
                                           (plate,
                                            edt_limit,
                                            segment_tag,
                                            go_tag,
                                           )
                                          )
                processes.append((plate, process))
        pool.close()
        pool.join()
        for plate, process in processes:
            attenuated_image = process.get()
            plate.image_stash[tag_out] = attenuated_image

    def measure_intensities(self,
                            segment_tag='cell_segment',
                            background_tag='image_background',
                            image_tag='sm_img',
                            husk_tag='husk_watershed_basins',
                            intensities_tag='basin_intensities',
                            multiplier=1,
                           ):
        #First find image background = everything outside the segmented cell.
        #
        #This is needed for measuring puncta intensities because we want to
        #measure intensity vs background of the cell, not its environment. We
        #add in the background below as part of the background_basins, which
        #are omitted by rp_intensity from background.
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            for RFP_plate, GFP_plate in izip(RFP_plates, GFP_plates):
                RFP_cell_segment = RFP_plate.image_stash[segment_tag]
                GFP_cell_segment = GFP_plate.image_stash[segment_tag]
                assert np.array_equal(RFP_cell_segment, GFP_cell_segment)
                image_background = label(~RFP_cell_segment)
                RFP_plate.image_stash[background_tag] = image_background
                GFP_plate.image_stash[background_tag] = image_background

        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            for plate in RFP_plates + GFP_plates:
                image = plate.image_stash[image_tag]
                basins = plate.feature_stash[husk_tag]
                image_background = plate.image_stash[background_tag]
                RP = regionprops(label_image=basins,
                                 intensity_image=image,
                                )
                intensities = {}
                background_basins = image_background + basins
                #Need to include all parts of the cell as background, no matter
                #where in the image they are.
                intensity_radius = max(image.shape)
                for rp in RP:
                    intensity = plate.rp_intensity(
                                           rp=rp,
                                           background=image,
                                           background_basins=background_basins,
                                           radius=intensity_radius,
                                           radius_factor=None,
                                           negative=False,
                                           multiplier=multiplier,
                                                  )
                    intensities[rp.label] = int(round(intensity))
                plate.feature_stash[intensities_tag] = intensities

    def build_3D_puncta(self,
                        husk_tag='husk_watershed_basins',
                        RFP_tag_out='RFP_label_graph',
                        GFP_tag_out='GFP_label_graph',
                       ):
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():

            if len(RFP_plates) == 0 or len(GFP_plates) == 0:
                raise RuntimeWarning(str(replicate) + " has "
                                     + str(len(RFP_plates)) 
                                     + " RFP_plates and "
                                     + str(len(GFP_plates))
                                     + " GFP_plates. Skipping."
                                    )
                continue

            RFP_label_graph = nx.Graph()

            for p, (plate_1, plate_2) in enumerate(Plate.pairwise(RFP_plates)):
                labels_1 = plate_1.feature_stash[husk_tag]
                labels_2 = plate_2.feature_stash[husk_tag]
                for (h, w), L in np.ndenumerate(labels_1):
                    if L != 0:
                        RFP_label_graph.add_node((p, 'RFP', L))
                for (h, w), L in np.ndenumerate(labels_2):
                    if L != 0:
                        RFP_label_graph.add_node((p + 1, 'RFP', L))
                labels_1_coordinates = defaultdict(set)
                for (h, w), L in np.ndenumerate(labels_1):
                    if L != 0:
                        labels_1_coordinates[L].add((h, w))
                labels_2_coordinates = defaultdict(set)
                for (h, w), L in np.ndenumerate(labels_2):
                    if L != 0:
                        labels_2_coordinates[L].add((h, w))
                for L1, coord_set_1 in labels_1_coordinates.iteritems():
                    for L2, coord_set_2 in labels_2_coordinates.iteritems():
                        if not coord_set_1.isdisjoint(coord_set_2):
                            RFP_label_graph.add_edge((p, 'RFP', L1),
                                                     (p + 1, 'RFP', L2)
                                                    )

            RFP_plates[0].feature_stash[RFP_tag_out] = RFP_label_graph

            GFP_label_graph = nx.Graph()

            for p, (plate_1, plate_2) in enumerate(Plate.pairwise(GFP_plates)):
                labels_1 = plate_1.feature_stash[husk_tag]
                labels_2 = plate_2.feature_stash[husk_tag]
                for (h, w), L in np.ndenumerate(labels_1):
                    if L != 0:
                        GFP_label_graph.add_node((p, 'GFP', L))
                for (h, w), L in np.ndenumerate(labels_2):
                    if L != 0:
                        GFP_label_graph.add_node((p + 1, 'GFP', L))
                labels_1_coordinates = defaultdict(set)
                for (h, w), L in np.ndenumerate(labels_1):
                    if L != 0:
                        labels_1_coordinates[L].add((h, w))
                labels_2_coordinates = defaultdict(set)
                for (h, w), L in np.ndenumerate(labels_2):
                    if L != 0:
                        labels_2_coordinates[L].add((h, w))
                for L1, coord_set_1 in labels_1_coordinates.iteritems():
                    for L2, coord_set_2 in labels_2_coordinates.iteritems():
                        if not coord_set_1.isdisjoint(coord_set_2):
                            GFP_label_graph.add_edge((p, 'GFP', L1),
                                                     (p + 1, 'GFP', L2)
                                                    )

            GFP_plates[0].feature_stash[GFP_tag_out] = GFP_label_graph

    def build_3D_puncta_MP(self,
                           husk_tag='husk_watershed_basins',
                           RFP_tag_out='RFP_label_graph',
                           GFP_tag_out='GFP_label_graph',
                           num_processes=None,
                          ):
        pool = multiprocessing.Pool(processes=num_processes,
                                    maxtasksperchild=None,
                                   )
        processes = []
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            if len(RFP_plates) == 0 or len(GFP_plates) == 0:
                raise RuntimeWarning(str(replicate) + " has "
                                     + str(len(RFP_plates))
                                     + " RFP_plates and "
                                     + str(len(GFP_plates))
                                     + " GFP_plates. Skipping."
                                    )
                continue
            RFP_process = pool.apply_async(build_3D_puncta,
                                           (RFP_plates,
                                            'RFP',
                                            husk_tag,
                                           )
                                          )
            GFP_process = pool.apply_async(build_3D_puncta,
                                           (GFP_plates,
                                            'GFP',
                                            husk_tag,
                                           )
                                          )
            processes.append((RFP_plates,
                              GFP_plates,
                              RFP_process,
                              GFP_process,
                             )
                            )
        pool.close()
        pool.join()
        for RFP_plates, GFP_plates, RFP_process, GFP_process in processes:
            RFP_label_graph = RFP_process.get()
            GFP_label_graph = GFP_process.get()
            RFP_plates[0].feature_stash[RFP_tag_out] = RFP_label_graph
            GFP_plates[0].feature_stash[GFP_tag_out] = GFP_label_graph

    def find_puncta_overlaps(self,
                             husk_tag='husk_watershed_basins',
                             RFP_label_tag='RFP_label_graph',
                             GFP_label_tag='GFP_label_graph',
                             tag_out='threeD_overlaps',
                            ):
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            if len(RFP_plates) == 0 or len(GFP_plates) == 0:
                raise RuntimeWarning(str(replicate) + " has "
                                     + str(len(RFP_plates))
                                     + " RFP_plates and "
                                     + str(len(GFP_plates))
                                     + " GFP_plates. Skipping."
                                    )
                continue

            for z, (RFP_plate, GFP_plate) in enumerate(izip(RFP_plates,
                                                            GFP_plates,
                                                           )
                                                      ):
                assert (RFP_plate.feature_stash['channel'] == 'RFP'
                        and GFP_plate.feature_stash['channel'] == 'GFP'
                       )
                RFP_basins = RFP_plate.feature_stash[husk_tag]
                GFP_basins = GFP_plate.feature_stash[husk_tag]
                RFP_label_graph = RFP_plates[0].feature_stash[RFP_label_tag]
                GFP_label_graph = GFP_plates[0].feature_stash[GFP_label_tag]
                R_label_set = set(RFP_basins.reshape(-1).tolist()) - set([0])
                G_label_set = set(GFP_basins.reshape(-1).tolist()) - set([0])
                for R_label in iter(R_label_set):
                    overlap_array = np.where(RFP_basins == R_label, 1, 0)
                    overlap_array = np.where(GFP_basins != 0, overlap_array, 0)
                    GFP_subimage = GFP_basins * overlap_array
                    GFP_overlapped_labels = \
                                        (set(GFP_subimage.reshape(-1).tolist())
                                         - set([0])
                                        )
                    overlaps = (z,
                                R_label,
                                overlap_array,
                                GFP_subimage,
                                GFP_overlapped_labels,
                               )
                    RFP_puncta = (z, 'RFP', R_label)
                    RFP_label_graph.node[RFP_puncta]['overlaps'] = overlaps
                for G_label in iter(G_label_set):
                    overlap_array = np.where(GFP_basins == G_label, 1, 0)
                    overlap_array = np.where(RFP_basins != 0, overlap_array, 0)
                    RFP_subimage = RFP_basins * overlap_array
                    RFP_overlapped_labels = \
                                        (set(RFP_subimage.reshape(-1).tolist())
                                         - set([0])
                                        )
                    overlaps = (z,
                                G_label,
                                overlap_array,
                                RFP_subimage,
                                RFP_overlapped_labels,
                               )
                    GFP_puncta = (z, 'GFP', G_label)
                    GFP_label_graph.node[GFP_puncta]['overlaps'] = overlaps

            RFP_label_graph = RFP_plates[0].feature_stash[RFP_label_tag]
            GFP_label_graph = GFP_plates[0].feature_stash[GFP_label_tag]
            threeD_RFP_overlaps = defaultdict(list)
            threeD_GFP_overlaps = defaultdict(list)
            threeD_RFP_puncta = nx.connected_components(RFP_label_graph)
            threeD_GFP_puncta = nx.connected_components(GFP_label_graph)
            for puncta in threeD_RFP_puncta:
                puncta_key = frozenset(puncta)
                assert puncta_key not in threeD_RFP_overlaps
                for (z, channel, Label) in iter(puncta):
                    assert channel == 'RFP'
                    overlaps = \
                          RFP_label_graph.node[(z, channel, Label)]['overlaps']
                    threeD_RFP_overlaps[puncta_key].append(overlaps)
            threeD_RFP_overlaps = {puncta_key: tuple(overlaps_list)
                                   for puncta_key, overlaps_list
                                   in threeD_RFP_overlaps.iteritems()
                                  }
            for puncta in threeD_GFP_puncta:
                puncta_key = frozenset(puncta)
                assert puncta_key not in threeD_GFP_overlaps
                for (z, channel, Label) in iter(puncta):
                    assert channel == 'GFP'
                    overlaps = \
                          GFP_label_graph.node[(z, channel, Label)]['overlaps']
                    threeD_GFP_overlaps[puncta_key].append(overlaps)
            threeD_GFP_overlaps = {puncta_key: tuple(overlaps_list)
                                   for puncta_key, overlaps_list
                                   in threeD_GFP_overlaps.iteritems()
                                  }
            RFP_plates[0].feature_stash[tag_out] = threeD_RFP_overlaps
            GFP_plates[0].feature_stash[tag_out] = threeD_GFP_overlaps

    def find_puncta_overlaps_MP(self,
                                husk_tag='husk_watershed_basins',
                                RFP_label_tag='RFP_label_graph',
                                GFP_label_tag='GFP_label_graph',
                                tag_out='threeD_overlaps',
                                num_processes=None,
                               ):
        pool = multiprocessing.Pool(processes=num_processes,
                                    maxtasksperchild=None,
                                   )
        processes = []
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            if len(RFP_plates) == 0 or len(GFP_plates) == 0:
                raise RuntimeWarning(str(replicate) + " has "
                                     + str(len(RFP_plates))
                                     + " RFP_plates and "
                                     + str(len(GFP_plates))
                                     + " GFP_plates. Skipping."
                                    )
                continue
            process = pool.apply_async(find_puncta_overlaps,
                                       (RFP_plates,
                                        GFP_plates,
                                        husk_tag,
                                        RFP_label_tag,
                                        GFP_label_tag,
                                       )
                                      )
            processes.append((RFP_plates, GFP_plates, process))
        pool.close()
        pool.join()
        for RFP_plates, GFP_plates, process in processes:
            threeD_RFP_overlaps, threeD_GFP_overlaps = process.get()
            RFP_plates[0].feature_stash[tag_out] = threeD_RFP_overlaps
            GFP_plates[0].feature_stash[tag_out] = threeD_GFP_overlaps

    @staticmethod
    def lookup_puncta(puncta_label_graph,
                      z,
                      channel,
                      Label,
                     ):
        all_puncta_keys = {frozenset(puncta_key): False
                           for puncta_key
                           in nx.connected_components(puncta_label_graph)
                          }
        for puncta_key in all_puncta_keys.iterkeys():
            for (z2, channel2, Label2) in iter(puncta_key):
                if z == z2 and channel == channel2 and Label == Label2:
                    all_puncta_keys[puncta_key] = True
                    break
        num_true = len([value for value in all_puncta_keys.values() if value])
        num_false = len([value for value in all_puncta_keys.values()
                         if not value
                        ]
                       )
        assert num_true + num_false == len(all_puncta_keys)
        assert num_true == 1
        assert num_false == len(all_puncta_keys) - 1
        which_key = [key
                     for key, value in all_puncta_keys.iteritems()
                     if value][0]
        return which_key

    def puncta_xcorr(self,
                     image_tag='sm_img',
                     RFP_label_tag='RFP_label_graph',
                     GFP_label_tag='GFP_label_graph',
                     tag_in='threeD_overlaps',
                     tag_out='xcorrs',
                     nonoverlapping_tag_out='nonoverlapping_puncta',
                    ):
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            if len(RFP_plates) == 0 or len(GFP_plates) == 0:
                raise RuntimeWarning(str(replicate) + " has "
                                     + str(len(RFP_plates))
                                     + " RFP_plates and "
                                     + str(len(GFP_plates))
                                     + " GFP_plates. Skipping."
                                    )
                continue

            threeD_RFP_overlaps = RFP_plates[0].feature_stash[tag_in]
            threeD_GFP_overlaps = GFP_plates[0].feature_stash[tag_in]
            RFP_label_graph = RFP_plates[0].feature_stash[RFP_label_tag]
            GFP_label_graph = GFP_plates[0].feature_stash[GFP_label_tag]

            RFP_xcorrs, GFP_xcorrs = defaultdict(float), defaultdict(float)
            overlapping_RFP_puncta, overlapping_GFP_puncta = set(), set()
            nonoverlapping_RFP_puncta, nonoverlapping_GFP_puncta = set(), set()

            for puncta_key, overlaps_tuple in threeD_RFP_overlaps.iteritems():
                assert puncta_key not in RFP_xcorrs
                assert puncta_key not in overlapping_RFP_puncta
                assert puncta_key not in nonoverlapping_RFP_puncta
                nonoverlapping_RFP_puncta.add(puncta_key)
                for overlaps in overlaps_tuple:
                    (z,
                     R_label,
                     overlap_array,
                     GFP_subimage,
                     GFP_overlapped_labels,
                    ) = overlaps
                    RFP_image = RFP_plates[z].image_stash[image_tag]
                    GFP_image = GFP_plates[z].image_stash[image_tag]
                    for G_label in iter(GFP_overlapped_labels):
                        G_overlap = np.where(GFP_subimage == G_label, 1, 0)
                        R_img = np.where(G_overlap, RFP_image, 0).reshape(-1)
                        G_img = np.where(G_overlap, GFP_image, 0).reshape(-1)
                        xc = np.correlate(G_img, R_img)[0]
                        other_puncta_key = PunctaExperiment.lookup_puncta(
                                            puncta_label_graph=GFP_label_graph,
                                            z=z,
                                            channel='GFP',
                                            Label=G_label,
                                                                         )
                        xc_key = frozenset([puncta_key, other_puncta_key])
                        overlapping_RFP_puncta.add(puncta_key)
                        RFP_xcorrs[xc_key] += xc
            nonoverlapping_RFP_puncta -= overlapping_RFP_puncta
            assert nonoverlapping_RFP_puncta.isdisjoint(overlapping_RFP_puncta)
            assert (overlapping_RFP_puncta
                    | nonoverlapping_RFP_puncta
                   ) == set(threeD_RFP_overlaps)

            for puncta_key, overlaps_tuple in threeD_GFP_overlaps.iteritems():
                assert puncta_key not in GFP_xcorrs
                assert puncta_key not in overlapping_GFP_puncta
                assert puncta_key not in nonoverlapping_GFP_puncta
                nonoverlapping_GFP_puncta.add(puncta_key)
                for overlaps in overlaps_tuple:
                    (z,
                     G_label,
                     overlap_array,
                     RFP_subimage,
                     RFP_overlapped_labels,
                    ) = overlaps
                    GFP_image = GFP_plates[z].image_stash[image_tag]
                    RFP_image = RFP_plates[z].image_stash[image_tag]
                    for R_label in iter(RFP_overlapped_labels):
                        R_overlap = np.where(RFP_subimage == R_label, 1, 0)
                        G_img = np.where(R_overlap, GFP_image, 0).reshape(-1)
                        R_img = np.where(R_overlap, RFP_image, 0).reshape(-1)
                        xc = np.correlate(R_img, G_img)[0]
                        other_puncta_key = PunctaExperiment.lookup_puncta(
                                            puncta_label_graph=RFP_label_graph,
                                            z=z,
                                            channel='RFP',
                                            Label=R_label,
                                                                         )
                        xc_key = frozenset([puncta_key, other_puncta_key])
                        overlapping_GFP_puncta.add(puncta_key)
                        GFP_xcorrs[xc_key] += xc
            nonoverlapping_GFP_puncta -= overlapping_GFP_puncta
            assert nonoverlapping_GFP_puncta.isdisjoint(overlapping_GFP_puncta)
            assert (overlapping_GFP_puncta
                    | nonoverlapping_GFP_puncta
                   ) == set(threeD_GFP_overlaps)

            error_tolerance = 10**-10
            for key in RFP_xcorrs.iterkeys():
                assert key in GFP_xcorrs
                error = abs(RFP_xcorrs[key] - GFP_xcorrs[key])
                assert error <= error_tolerance

            RFP_plates[0].feature_stash[tag_out] = RFP_xcorrs
            GFP_plates[0].feature_stash[tag_out] = GFP_xcorrs
            RFP_plates[0].feature_stash[nonoverlapping_tag_out] = nonoverlapping_RFP_puncta
            GFP_plates[0].feature_stash[nonoverlapping_tag_out] = nonoverlapping_GFP_puncta

    def segment_all_cells_3D(self,
                             use_channel='GFP',
                             selem_size=5,
                             collapse_threshold=2,
                             image_tag='go_image',
                             tag_out='cell_segment',
                             omit_replicates=None,
                             num_processes=None,
                            ):
        pool = multiprocessing.Pool(processes=num_processes,
                                    maxtasksperchild=None,
                                   )
        processes = []
        if omit_replicates is None:
            omit_replicates = set()
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            if replicate in omit_replicates:
                continue
            if use_channel == 'GFP':
                plates = GFP_plates
            else:
                plates = RFP_plates
            process = pool.apply_async(segment_cell_3D,
                                       (plates,
                                        selem_size,
                                        collapse_threshold,
                                        image_tag,
                                       )
                                      )
            processes.append((RFP_plates, GFP_plates, process))
        pool.close()
        pool.join()
        for RFP_plates, GFP_plates, process in processes:
            segments = process.get()
            for z, segment in enumerate(segments):
                RFP_plates[z].image_stash[tag_out] = segment
                GFP_plates[z].image_stash[tag_out] = segment

    def make_husks_MP(self,
                      image_tag='go_image',
                      segment_tag='cell_segment',
                      mf_tag_out='mf_img',
                      sm_tag_out='sm_img',
                      co_tag_out='co_img',
                      median_size=51,
                      selem_size=2,
                      median_fill=False,
                      omit_replicates=None,
                      null_segment=False,
                      num_processes=None,
                     ):
        pool = multiprocessing.Pool(processes=num_processes,
                                    maxtasksperchild=None,
                                   )
        processes = []
        if omit_replicates is None:
            omit_replicates = set()
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            if replicate in omit_replicates:
                continue
            for plate in RFP_plates + GFP_plates:
                process = pool.apply_async(make_husk,
                                           (plate,
                                            image_tag,
                                            segment_tag,
                                            median_size,
                                            selem_size,
                                            median_fill,
                                            null_segment,
                                           )
                                          )
                processes.append((plate, process))
        pool.close()
        pool.join()
        for plate, process in processes:
            mf_img, sm_img, co_img = process.get()
            plate.image_stash[mf_tag_out] = mf_img
            plate.image_stash[sm_tag_out] = sm_img
            plate.image_stash[co_tag_out] = co_img

    def segment_again_MP(self,
                         use_channel='GFP',
                         selem_size=5,
                         image_tag='go_image',
                         segment_tag='cell_segment',
                         tag_out='second_segment',
                         omit_replicates=None,
                         num_processes=None,
                        ):
        pool = multiprocessing.Pool(processes=num_processes,
                                    maxtasksperchild=None,
                                   )
        processes = []
        if omit_replicates is None:
            omit_replicates = set()
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            if replicate in omit_replicates:
                continue
            for RFP_plate, GFP_plate in izip(RFP_plates, GFP_plates):
                if use_channel == 'GFP':
                    plate = GFP_plate
                else:
                    plate = RFP_plate
                process = pool.apply_async(segment_again,
                                           (plate,
                                            selem_size,
                                            image_tag,
                                            segment_tag,
                                           )
                                          )
                processes.append((RFP_plate, GFP_plate, process))
        pool.close()
        pool.join()
        for RFP_plate, GFP_plate, process in processes:
            second_segment = process.get()
            RFP_plate.image_stash[tag_out] = second_segment
            GFP_plate.image_stash[tag_out] = second_segment

    def puncta_overlap_metric(self,
                              tag_in='threeD_overlaps',
                              tag_out='overlap_metric',
                              normalize=False,
                              husk_tag='husk_watershed_basins',
                             ):
        for replicate, (RFP_plates, GFP_plates) in self.replicates.iteritems():
            if len(RFP_plates) == 0 or len(GFP_plates) == 0:
                raise RuntimeWarning(str(replicate) + " has "
                                     + str(len(RFP_plates))
                                     + " RFP_plates and "
                                     + str(len(GFP_plates))
                                     + " GFP_plates. Skipping."
                                    )
                continue           

            threeD_RFP_overlaps = RFP_plates[0].feature_stash[tag_in]
            threeD_GFP_overlaps = GFP_plates[0].feature_stash[tag_in]

            if normalize:
                RFP_overlap_metric, GFP_overlap_metric = {}, {}
            else:
                RFP_overlap_metric = defaultdict(float)
                GFP_overlap_metric = defaultdict(float)

            for puncta_key, overlaps_tuple in threeD_RFP_overlaps.iteritems():
                for overlaps in overlaps_tuple:
                    (z,
                     R_label,
                     overlap_array,
                     GFP_subimage,
                     GFP_overlapped_labels,
                    ) = overlaps
                    if normalize:
                        numerator = np.sum(overlap_array)
                        RFP_plate = RFP_plates[z]
                        RFP_basins = RFP_plate.feature_stash[husk_tag]
                        Label_area = np.sum(np.where(RFP_basins == R_label,
                                                     1,
                                                     0,
                                                    )
                                           )
                        denominator = Label_area
                        accumulator = RFP_overlap_metric.setdefault(puncta_key,
                                                                    [0, 0],
                                                                   )
                        accumulator[0] += numerator
                        accumulator[1] += denominator
                    else:
                        RFP_overlap_metric[puncta_key] += np.sum(overlap_array)

            for puncta_key, overlaps_tuple in threeD_GFP_overlaps.iteritems():
                for overlaps in overlaps_tuple:
                    (z,
                     G_label,
                     overlap_array,
                     RFP_subimage,
                     RFP_overlapped_labels,
                    ) = overlaps
                    if normalize:
                        numerator = np.sum(overlap_array)
                        GFP_plate = GFP_plates[z]
                        GFP_basins = GFP_plate.feature_stash[husk_tag]
                        Label_area = np.sum(np.where(GFP_basins == G_label,
                                                     1,
                                                     0,
                                                    )
                                           )
                        denominator = Label_area
                        accumulator = GFP_overlap_metric.setdefault(puncta_key,
                                                                    [0, 0],
                                                                   )
                        accumulator[0] += numerator
                        accumulator[1] += denominator
                    else:
                        GFP_overlap_metric[puncta_key] += np.sum(overlap_array)

            if normalize:
                RFP_overlap_metric = {puncta_key: (float(accumulator[0])
                                                   / 
                                                   accumulator[1]
                                                  )
                                      for puncta_key, accumulator
                                      in RFP_overlap_metric.iteritems()
                                     }
                GFP_overlap_metric = {puncta_key: (float(accumulator[0])
                                                   /
                                                   accumulator[1]
                                                  )
                                      for puncta_key, accumulator
                                      in GFP_overlap_metric.iteritems()
                                     }

            RFP_plates[0].feature_stash[tag_out] = RFP_overlap_metric
            GFP_plates[0].feature_stash[tag_out] = GFP_overlap_metric

    @staticmethod
    def hw_bbox(segment):
        B = np.argwhere(segment)
        (hmin, wmin), (hmax, wmax) = B.min(0), B.max(0)
        return hmin, wmin, hmax, wmax

    @staticmethod
    def is_inside_segment(label_mask,
                          segment,
                         ):
        overlap = label_mask * segment
        negative_overlap = ~overlap
        outside = negative_overlap * label_mask
        if np.any(outside):
            return False
        else:
            return True

    @staticmethod
    def shift_puncta_mask(puncta_mask,
                          h_offset,
                          w_offset,
                          z_offset,
                          absolute=True,
                         ):
        shifted_puncta_mask = [np.zeros_like(mask) for mask in puncta_mask]
        for z, mask in enumerate(puncta_mask):
            if z + z_offset >= len(shifted_puncta_mask):
                continue
            if not np.any(mask):
                continue
            if absolute:
                hmin, wmin, hmax, wmax = PunctaExperiment.hw_bbox(segment=mask)
                hpos, wpos = float(hmin + hmax) / 2.0, float(wmin + wmax) / 2.0
                use_h_offset = h_offset - hpos
                use_w_offset = w_offset - wpos
                print((use_h_offset, use_w_offset))
            else:
                use_h_offset, use_w_offset = h_offset, w_offset
            shifted_mask = shift(mask,
                                 (use_h_offset, use_w_offset),
                                 cval=False,
                                )
            shifted_puncta_mask[z + z_offset] = shifted_mask
        return shifted_puncta_mask

    @staticmethod
    def is_inside_cell(puncta_mask,
                       segments,
                       h_offset,
                       w_offset,
                       z_offset,
                       absolute=True,
                      ):
        shifted_puncta_mask = PunctaExperiment.shift_puncta_mask(
                                                       puncta_mask=puncta_mask,
                                                       h_offset=h_offset,
                                                       w_offset=w_offset,
                                                       z_offset=z_offset,
                                                       absolute=absolute,
                                                                )
        if all([PunctaExperiment.is_inside_segment(label_mask=mask,
                                                   segment=segments[z],
                                                  )
                for z, mask in enumerate(shifted_puncta_mask)]):
            inside = True
        else:
            inside = False
            shifted_puncta_mask = None
        return inside, shifted_puncta_mask

    @staticmethod
    def make_puncta_mask(plates,
                         label_graph,
                         channel,
                         basins_tag,
                        ):
        puncta_mask = [np.zeros_like(plate.feature_stash[basins_tag],
                                     dtype=np.bool,
                                    )
                       for plate in plates
                      ]
        z_layers = []
        for z, chan, Label in label_graph.nodes():
            assert chan == channel
            layer_Labels = plates[z].feature_stash[basins_tag]
            layer_mask = np.where(layer_Labels == Label, True, False)
            puncta_mask[z] = np.logical_or(layer_mask.astype(np.bool),
                                           puncta_mask[z],
                                          )
            z_layers.append(z)
        z_layers = sorted(z_layers)
        assert all([z2 - z1 == 0 or z2 - z1 == 1
                    for z1, z2 in Plate.pairwise(z_layers)])
        assert len(plates) - 1 >= z_layers[-1] >= z_layers[0] >= 0
        return puncta_mask, z_layers

    @staticmethod
    def delimit_offset_boundaries(plates,
                                  z_layers,
                                  segments,
                                 ):
        zmin, zmax = -z_layers[0], len(plates) - 1 - z_layers[-1]
        assert zmin <= 0 <= zmax
        bounds = [PunctaExperiment.hw_bbox(segment=segment)
                  for segment in segments if np.any(segment)]
        hmin = min([hmin for hmin, wmin, hmax, wmax in bounds])
        wmin = min([wmin for hmin, wmin, hmax, wmax in bounds])
        hmax = min([hmax for hmin, wmin, hmax, wmax in bounds])
        wmax = min([wmax for hmin, wmin, hmax, wmax in bounds])
        return hmin, wmin, zmin, hmax, wmax, zmax
    
    @staticmethod
    def place_puncta(plates,
                     channel,
                     puncta,
                     segment_tag='second_segment',
                     basins_tag='husk_watershed_basins',
                     max_tries=10**3,
                     existing_basins_tag=None,
                     segment_placement=False,
                     force_h_offset=None,
                     force_w_offset=None,
                     force_z_offset=None,
                    ):
        puncta_mask, z_layers = PunctaExperiment.make_puncta_mask(
                                                         plates=plates,
                                                         label_graph=puncta,
                                                         channel=channel,
                                                         basins_tag=basins_tag,
                                                                 )
        segments = [plate.image_stash[segment_tag] for plate in plates]
        hmin, wmin, zmin, hmax, wmax, zmax = \
                  PunctaExperiment.delimit_offset_boundaries(plates=plates,
                                                             z_layers=z_layers,
                                                             segments=segments,
                                                            )
        h_size, w_size = plates[0].feature_stash[basins_tag].shape
        placed_mask = None
        placed_h, placed_w, placed_z = None, None, None
        p_hmin, p_wmin, p_zmin, p_hmax, p_wmax, p_zmax = \
               PunctaExperiment.delimit_offset_boundaries(plates=plates,
                                                          z_layers=z_layers,
                                                          segments=puncta_mask,
                                                         )
        p_hpos = int(round(float(p_hmin + p_hmax) / 2.0))
        p_wpos = int(round(float(p_wmin + p_wmax) / 2.0))

        if segment_placement:
            possible_coordinates = []
            for z, segment in enumerate(segments):
                possible_hws = zip(*np.where(segment))
                for h, w in possible_hws:
                    possible_coordinates.append((h, w, z))
            #shuffle is problematic for large numbers of coordinates because
            #random number generator period is too short
            #shuffle(possible_coordinates)

        while max_tries > 0:
            max_tries -= 1
            if segment_placement:
                h_offset, w_offset, z_offset = choice(possible_coordinates)
                z_offset -= z_layers[0]
                h_offset -= p_hpos
                w_offset -= p_wpos
            else:
                z_offset = randint(zmin, zmax)
                h_offset = randint(int(floor(hmin - p_hpos)),
                                   int(ceil(hmax + 1 - p_hpos)),
                                  )
                w_offset = randint(int(floor(wmin - p_wpos)),
                                   int(ceil(wmax + 1 - p_wpos)),
                                  )
            if (force_h_offset is not None
                or force_w_offset is not None
                or force_z_offset is not None
               ):
                if (force_h_offset is None
                    or force_w_offset is None
                    or force_z_offset is None
                   ):
                    raise ValueError("All or nothing.")
                h_offset = force_h_offset
                w_offset = force_w_offset
                z_offset = force_z_offset
            inside, shifted_puncta_mask = \
                       PunctaExperiment.is_inside_cell(puncta_mask=puncta_mask,
                                                       segments=segments,
                                                       h_offset=h_offset,
                                                       w_offset=w_offset,
                                                       z_offset=z_offset,
                                                       absolute=False,
                                                      )

            if not inside:
                continue
            if existing_basins_tag is not None:
                existing_basins = [plate.feature_stash[existing_basins_tag]
                                   for plate in plates
                                  ]
                overlaps = [mask * basins
                            for mask, basins
                            in izip(shifted_puncta_mask, existing_basins)
                           ]
                if any([np.any(overlap) for overlap in overlaps]):
                    continue
            placed_mask = shifted_puncta_mask
            placed_h, placed_w, placed_z = h_offset, w_offset, z_offset
            break
        return placed_mask, placed_h, placed_w, placed_z, max_tries

    @staticmethod
    def generate_basins(puncta,
                        plates,
                        channel,
                        placed_h,
                        placed_w,
                        placed_z,
                        basins_tag='husk_watershed_basins',
                        tag_out='synthetic_basins',
                        add=False,
                       ):
        synthetic_basins = [np.zeros_like(plate.feature_stash[basins_tag])
                            for plate in plates
                           ]
        for z, chan, Label in puncta.nodes():
            if chan != channel:
                raise ValueError(str(chan) + " != " + str(channel))
            if z + placed_z >= len(plates):
                continue
            assert Label != 0
            basins = plates[z].feature_stash[basins_tag]
            #Shifting integer label may cause unwanted interpolation??
            basins = np.where(basins == Label, Label, 0)
            #basins = np.where(basins == Label, True, False)
            shifted_basins = shift(basins,
                                   (placed_h, placed_w),
                                   cval=False,
                                   order=0,
                                  )
            #shifted_basins = np.where(shifted_basins == True, Label, 0)
            assert not np.any(shifted_basins * synthetic_basins[z + placed_z])
            synthetic_basins[z + placed_z] += shifted_basins
        for z, plate in enumerate(plates):
            if add:
                existing_basins = plate.feature_stash[tag_out]
                highest_index = np.amax(existing_basins) + 1
                assert highest_index == int(highest_index)
                assert highest_index > 0
                if np.any(synthetic_basins[z] * plate.feature_stash[tag_out]):
                    debug_intersection = (
                                  set(zip(*np.where(synthetic_basins[z] != 0)))
                                  & set(zip(*np.where(plate.feature_stash[tag_out] != 0)))
                                         )                                      
                    print("debug_intersection = " + str(debug_intersection))
                    intersection_values = \
                                        [(synthetic_basins[z][X, Y],
                                          plate.feature_stash[tag_out][X, Y],
                                         )
                                         for (X, Y) in iter(debug_intersection)
                                        ]
                    print(intersection_values)
                    print("cleaning up")
                    for (X, Y) in iter(debug_intersection):
                        synthetic_basins[z][X, Y] = 0
                highest_index_mask = np.where(synthetic_basins[z] != 0,
                                              highest_index,
                                              0,
                                             )
                assert not np.any(synthetic_basins[z]
                                  * plate.feature_stash[tag_out]
                                 ), (
                    set(zip(*np.where(synthetic_basins[z] != 0)))
                    & set(zip(*np.where(plate.feature_stash[tag_out] != 0)))
                                    )
                assert not np.any(highest_index_mask *
                                  plate.feature_stash[tag_out]
                                 )
                plate.feature_stash[tag_out] += (synthetic_basins[z]
                                                 + highest_index_mask
                                                )
            else:
                plate.feature_stash[tag_out] = synthetic_basins[z]
