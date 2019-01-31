#!/usr/bin/env python2

#Need matplotlib for image display
import matplotlib
import matplotlib.pyplot as plt
import xlsxwriter
import argparse
import collections
from sys import argv

#Import other Python libraries we use
from glob import glob
from datetime import datetime
from scipy.misc import imread

#Import image analysis library
import solenodon

#Looks like basins #196, #280 wasn't divided properly
#We'll ask the computer to try again
#This code hasn't been merged into the main library yet

import numpy as np
from skimage.feature import peak_local_max
from skimage.color import rgb2gray
from skimage.measure import label
from skimage.morphology import watershed
from skimage.io import imsave

def subdivide_basin(basins,
                    target_basin,
                    grayscale_image,
                    smoothing_sigma=None,
                   ):
    if smoothing_sigma is not None:
        grayscale_image = gaussian(grayscale_image, sigma=smoothing_sigma)
    n_img = np.amax(grayscale_image) - grayscale_image
    maxima_distance = 5
    local_maxima = peak_local_max(n_img,
                                  indices=False,
                                  min_distance=maxima_distance,
                                 )
    markers = label(local_maxima)
    W_labels = watershed(grayscale_image,
                         markers=markers,
                        )
    W_labels = np.where(basins == target_basin,
                        W_labels,
                        0,
                       )
    return W_labels

def subdivide_basins(basins,
                     target_basins,
                     grayscale_image,
                     smoothing_sigma=None,
                     all_basins=True,
                    ):
    subdivided_labels = [subdivide_basin(basins=basins,
                                         target_basin=basin,
                                         grayscale_image=grayscale_image,
                                         smoothing_sigma=None,
                                        )
                         for basin in iter(target_basins)]
    if all_basins:
        for basin in iter(target_basins):
            basins = np.where(basins == basin, 0, basins)
    else:
        basins = np.zeros_like(basins)
    for labels in subdivided_labels:
        basins += labels
    return basins

#Initialize Plate objects to hold each image
def main():
    #target_plates lets us choose which file(s) to analyze if there's many of them
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image")
    args = vars(ap.parse_args())
    print(args["image"])

    target_plates = args['image']
    '''

    plates = []
    failed_plates = []

    # create workbook and sheet
    workbook = xlsxwriter.Workbook("{}.xlsx".format("test"))
    worksheet = workbook.add_worksheet()

    for f in sorted(glob("*.*")):

        if not f.endswith(('.png', '.jpg', '.jpeg')):

            continue
        try:
            plate = solenodon.Plate(image=imread(f),
                                    tag_in='original_image',
                                    source_filename=f,
                                   )
            plates.append(plate)

        except Exception as e:
            print(e)
            print(plate.metadata['source_filename'])
            failed_plates.add(plate)
    print("Files loaded at " + str(datetime.now()))

    # Crop out empty space around plates
    for p, plate in enumerate(plates):
        try:
            plate.crop_to_plate(tag_in='original_image',
                                tag_out='cropped_image',
                                feature_out='crop_rotation',
                                )
        except Exception as e:
            print(e)
            print(plate.metadata['source_filename'])
            failed_plates.add(plate)
    print("Plates cropped at " + str(datetime.now()))

    # Rescale image to standard size
    # This is very important because the image morphology parameters we use for analysis are defined
    # in terms of pixels and therefore are specific to a (ballpark) resolution.

    for plate in plates:
        plate.rescale_image(tag_in='cropped_image',
                            tag_out='rescaled_image',
                            target_height=500,
                            )

    # Trim the outermost pixels a bit to make sure no background remains around the edges

    for p, plate in enumerate(plates):
        try:
            plate.crop_border(tag_in='rescaled_image',
                              tag_out='border_cropped_image',
                              border=20,
                              )
        except Exception as e:
            print(e)
            print(plate.metadata['source_filename'])
            failed_plates.add(plate)
    print("Borders cropped at " + str(datetime.now()))

    # Median correct the image to correct uneven intensity over the plate
    # Display the result so we see what we have so far

    for plate in plates:
        uncorrected_image = plate.image_stash['border_cropped_image']
        corrected_image = solenodon.Plate.median_correct_image(image=uncorrected_image,
                                                               median_disk_radius=31,
                                                               )
        plate.image_stash['corrected_border_cropped_image'] = corrected_image
        plate.display(tag_in='corrected_border_cropped_image',
                      figsize=20,
                      output_filename = "cropped_image.jpg"
                      )

    # Let's try segmenting the spots using the waterfall algorithm

    for plate in plates:
        plate.waterfall_segmentation(tag_in='corrected_border_cropped_image',
                                     feature_out='waterfall_basins',
                                     R_out='R_img',
                                     mg_out='mg_img',
                                     median_disk_radius=31,
                                     smoothing_sigma=2,
                                     threshold_opening_size=2,
                                     basin_open_close_size=5,
                                     skeleton_label=0,
                                     debug_output=False,
                                     )

    # The largest item found is the background; we need to get rid of it
    for plate in plates:
        plate.remove_most_frequent_label(basins_feature='waterfall_basins',
                                         feature_out='filtered_waterfall_basins',
                                         debug_output=False,
                                         )

    # This performs an overlay of the smoothed lines on the original images
    # this case it mostly does nothing

    for plate in plates:
        plate.overlay_watershed(tag_in='corrected_border_cropped_image',
                                intensity_image_tag='corrected_border_cropped_image',
                                median_radius=None,
                                filter_basins=True,
                                waterfall_basins_feature='filtered_waterfall_basins',
                                feature_out='overlaid_watershed_basins',
                                min_localmax_dist=5,
                                smoothing_sigma=1,
                                min_area=10,
                                min_intensity=0.1,
                                # min_intensity=None,
                                # rp_radius_factor=0.5,
                                rp_radius_factor=None,
                                # rp_radius_factor=1.0,
                                # rp_radius_factor=2.0,
                                debug_output=False,
                                basin_open_close_size=None,
                                )
    # Measure basins

    for plate in plates:
        plate.measure_basin_intensities(tag_in='corrected_border_cropped_image',
                                        median_radius=None,
                                        filter_basins=True,
                                        radius_factor=None,
                                        basins_feature='overlaid_watershed_basins',
                                        feature_out='basin_intensities',
                                        multiplier=10.0
                                        )
        plate.find_basin_centroids(tag_in='corrected_border_cropped_image',
                                   basins_feature='overlaid_watershed_basins',
                                   feature_out='basin_centroids',
                                   )

    # Each spot is given a unique integer identifier, and its intensity is shown as I=

    '''
    for plate in plates:
        plate.display(tag_in='border_cropped_image',
                      figsize=70,
                      basins_feature='overlaid_watershed_basins',
                      basin_alpha=0.1,
                      baseline_feature=None,
                      solvent_front_feature=None,
                      lanes_feature=None,
                      basin_centroids_feature='basin_centroids',
                      basin_lane_assignments_feature=None,
                      # basin_intensities_feature='basin_intensities',
                      basin_rfs_feature=None,
                      lines_feature=None,
                      draw_boundaries=True,
                      side_by_side=False,
                      display_labels=True,
                      output_filename = "cropped_image.jpg"
                      )
    '''

    subdivided_basins = subdivide_basins(basins=plates[0].feature_stash['overlaid_watershed_basins'],
                                         target_basins=[197,
                                                        291,

                                                        ],
                                         grayscale_image=rgb2gray(plates[0].image_stash['border_cropped_image']),
                                         )
    # Manually add found basin

    plates[0].feature_stash['subdivided_basins'] = subdivided_basins

    for plate in plates:
        plate.measure_basin_intensities(tag_in='corrected_border_cropped_image',
                                        median_radius=None,
                                        filter_basins=True,
                                        radius_factor=None,
                                        basins_feature='subdivided_basins',
                                        feature_out='subdivided_basin_intensities',
                                        multiplier=10.0,
                                        )
        plate.find_basin_centroids(tag_in='corrected_border_cropped_image',
                                   basins_feature='subdivided_basins',
                                   feature_out='subdivided_basin_centroids',
                                   )

    image = plates[0].display(tag_in='border_cropped_image',
                      figsize=70,
                      basins_feature='subdivided_basins',
                      basin_alpha=0.1,
                      baseline_feature=None,
                      solvent_front_feature=None,
                      lanes_feature=None,
                      basin_centroids_feature='subdivided_basin_centroids',
                      basin_lane_assignments_feature=None,
                      basin_intensities_feature='subdivided_basin_intensities',
                      basin_rfs_feature=None,
                      lines_feature=None,
                      draw_boundaries=True,
                      side_by_side=False,
                      display_labels=True,
                      output_filename='analyzed_image.png'
                      )



    # Save basin intensities to excel

    row = 0
    column = 0

    values = plates[0].feature_stash['subdivided_basin_intensities']
    values = collections.OrderedDict(sorted(values.items()))
    worksheet.write(row, column, "basin")
    worksheet.write(row, column+1, "intensity")

    for basin in values:
        row += 1
        column = 0
        worksheet.write(row, column, basin)
        column += 1
        worksheet.write(row, column, values[basin])

    workbook.close()

main ()