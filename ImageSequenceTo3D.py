import numpy as np
import cv2
import argparse
import glob
import os
from scipy import signal
import sys
import open3d as o3d

def blur3d(voxels, radius=4, sigma=1.0, threshold=0.25):

	r = np.arange(-radius+1, radius, 1)
	xx, yy, zz = np.meshgrid(r, r, r)
	kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
	kernel /= kernel.sum()

	filtered = signal.convolve(voxels, kernel, mode="same")

	return filtered > threshold

def images_to_voxels(folder_path):
	# Find all images, assuming all items in path are images

	file_paths = glob.glob(os.path.join(folder_path, '*'))
	file_paths = sorted(file_paths)

	num_images = len(file_paths)

	# Allocate a numpy array assuming each image has the same dimensions as the first
	# Grab first image dimensions
	img_shape = cv2.imread(file_paths[0]).shape[:2]
	voxels = np.zeros((num_images, img_shape[0], img_shape[1]), dtype=bool)

	print()
	for i, file_path in enumerate(file_paths):
		# Load each file
		img = cv2.imread(file_path)

		voxels[i, :, :] = img.sum(axis=2) > 0

		if i % 25 == 0:
			sys.stdout.write(f'Image {i} of {len(file_paths)}                   \r')

	return voxels

def generate_possible_vertex_indices(shape):
	# Assume three vertices per voxel, excess vertices along -y, -z, +x far planes will be assigned but never used
	vertices_shape = (shape[0], shape[1], shape[2], 3)
	vertices_size = shape[0] * shape[1] * shape[2] * 3
	vertex_indices = np.arange(vertices_size, dtype=np.int32).reshape(vertices_shape)

	return vertex_indices

def generate_marching_cubes_offsets(shape):
	# Here we will create all possible triangle combinations for the marching cubes algorithm.
	#       +z
	#        ^   +y
	#        |  /
	#        | /
	#
	#        a-----0------b      -->  +x
	#       /|           /|
	#      2 |          6 |
	#     /  1         /  4
	#    d-----3------c   |
	#    |   |        |   |
	#    |   e-----8--|---f
	#    7  /         9  /
	#-   | 5          | 10
	#    |/           |/
	#    h-----11-----g

	# Table generated from here:
	# https://cg.informatik.uni-freiburg.de/intern/seminar/surfaceReconstruction_survey%20of%20marching%20cubes.pdf

	unique_case_table = np.array(
		[                                                                                          # Character sequence     bitfield equivalent
			[   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   ], #                        0
			[   [11,  5,  7],   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   ], # h                      128
			[   [10,  5,  7],   [ 9, 10,  7],   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   ], # hg                     192
			[   [11,  5,  7],   [ 3,  6,  9],   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   ], # hc                     132
			[   [11,  5,  7],   [ 6,  0,  4],   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   ], # hb                     130
			[   [11,  9,  5],   [ 9,  1,  5],   [ 9,  4,  1],   [-1, -1, -1],   [-1, -1, -1],   ], # gfe                    112
			[   [10,  5,  7],   [ 9, 10,  7],   [ 6,  0,  4],   [-1, -1, -1],   [-1, -1, -1],   ], # hgb                    194
			[   [ 2,  3,  7],   [ 6,  0,  4],   [ 9, 10, 11],   [-1, -1, -1],   [-1, -1, -1],   ], # dgb                    74
			[   [ 4,  7,  9],   [ 4,  1,  7],   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   ], # hgfe                   240
			[   [ 2,  7, 11],   [ 2, 11,  0],   [10,  0, 11],   [ 4,  0, 10],   [-1, -1, -1],   ], # hafe                   177
			[   [ 0,  2,  5],   [ 0,  5,  8],   [ 3,  6, 11],   [11,  6, 10],   [-1, -1, -1],   ], # aceg                   85
			[   [ 1,  7, 11],   [ 6,  1, 11],   [ 6, 11, 10],   [ 0,  1,  6],   [-1, -1, -1],   ], # hefb                   178
			[   [ 2,  3,  7],   [11,  9,  5],   [ 9,  1,  5],   [ 9,  4,  1],   [-1, -1, -1],   ], # gfed                   120
			[   [ 2,  3,  7],   [ 8,  1,  5],   [11,  9, 10],   [ 6,  0,  4],   [-1, -1, -1],   ], # bdeg                   90
			[   [ 2,  5, 11],   [ 2, 11,  4],   [ 0,  2,  4],   [ 9,  4, 11],   [-1, -1, -1],   ], # eagf                   113
			[   [ 2,  7, 11],   [ 0,  2, 11],   [ 0, 11, 10],   [ 9,  3,  6],   [ 0, 10,  4],   ], # acefh                  181
			[   [ 7,  6,  9],   [ 0,  6,  7],   [ 4,  0,  5],   [10,  4,  5],   [ 5,  0,  7],   ], # acdef                  61
			[   [ 9, 11,  5],   [ 1,  9,  5],   [ 4,  9,  1],   [-1, -1, -1],   [-1, -1, -1],   ], # abcdh                  143
			[   [ 5, 11,  7],   [ 0,  6,  4],   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   ], # acdefg                 125
			[   [ 3,  7,  5],   [ 3,  5,  6],   [ 9,  6,  5],   [11,  9,  5],   [-1, -1, -1],   ], # abdefg                 123
			[   [ 9,  7,  5],   [10,  9,  5],   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   ], # abcdef                 63
			[   [ 5, 11,  7],   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   ], # abcdefg                127
			[   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   [-1, -1, -1],   ], # abcdefgh               255
		],
		dtype=np.int32
	)

	unique_case_bitfield = np.array( # Same as bitfield table above
		[ 0, 128, 192, 132, 130, 112, 194, 74, 240, 177, 85, 178, 120, 90, 113, 181, 61, 143, 125, 123, 63, 127, 255 ],
		dtype=np.uint8
	)


	# Quick function to apply transformations defined in the maps below
	def perform_bitfield_transformation(old_bitfield_array, transformation):
		new_bitfield_array = np.zeros_like(old_bitfield_array)
		for c in 'abcdefgh':
			find = 1 << (ord(c) - ord('a'))
			replace = 1 << (ord(transformation[c]) - ord('a'))

			mask = np.bitwise_and(old_bitfield_array, find).astype(bool)
			new_bitfield_array[mask] += replace
		return new_bitfield_array

	def perform_index_transformation(indices, transformation, reverse_normal=True):
		orig_shape = indices.shape

		if reverse_normal:
			indices = indices[:, :, ::-1] # If we are reflecting, things that were once clockwise are now counter-clockwise, and that screws our normals, so reversing the order here can fix that

		flat_indices = indices.reshape(-1)
		replaced_flat_indices = transformation[flat_indices]
		replaced_indices = replaced_flat_indices.reshape(orig_shape)

		return replaced_indices

	# A map that represent how to roll the bitfield around ray ag
	roll_bitfield_transformation = {
		'a': 'a',
		'b': 'd',
		'd': 'e',
		'e': 'b',
		'c': 'h',
		'h': 'f',
		'f': 'c',
		'g': 'g',
	}
	roll_indices_transformation = np.array([ # Another map to roll the indices in the unique_case_table
		2,       # 0  -> 2
		0,       # 1  -> 0
		1,       # 2  -> 1
		5,       # 3  -> 5
		3,       # 4  -> 3
		4,       # 5  -> 4
		7,       # 6  -> 7
		8,       # 7  -> 8
		6,       # 8  -> 6
		11,      # 9  -> 11
		9,       # 10 -> 9
		10,      # 11 -> 10
		-1,      # The last element (the -1 element, needs to map to -1)
	])

	# Map to reflect across x axis
	x_reflect_bitfield_transformation = {
		'a': 'b',
		'c': 'd',
		'e': 'f',
		'g': 'h',

		'b': 'a',
		'd': 'c',
		'f': 'e',
		'h': 'g',
	}
	x_reflect_indices_transformation = np.array([
		0,       # 0  -> 0
		4,       # 1  -> 4
		6,       # 2  -> 6
		3,       # 3  -> 3
		1,       # 4  -> 1
		10,      # 5  -> 10
		2,       # 6  -> 2
		9,       # 7  -> 9
		8,       # 8  -> 8
		7,       # 9  -> 7
		5,       # 10 -> 5
		11,      # 11 -> 11
		-1,      # The last element (the -1 element, needs to map to -1)
	])

	# Reflect across y axis
	y_reflect_bitfield_transformation = {
		'a': 'd',
		'b': 'c',
		'e': 'h',
		'f': 'g',

		'd': 'a',
		'c': 'b',
		'h': 'e',
		'g': 'f',
	}
	y_reflect_indices_transformation = np.array([
		3,       # 0  -> 3
		7,       # 1  -> 7
		2,       # 2  -> 2
		0,       # 3  -> 0
		9,       # 4  -> 9
		5,       # 5  -> 5
		6,       # 6  -> 6
		1,       # 7  -> 1
		11,      # 8  -> 11
		4,       # 9  -> 4
		10,      # 10 -> 10
		8,       # 11 -> 8
		-1,      # The last element (the -1 element, needs to map to -1)
	])


	# Reflect across z axis
	z_reflect_bitfield_transformation = {
		'a': 'e',
		'b': 'f',
		'c': 'g',
		'd': 'h',

		'e': 'a',
		'f': 'b',
		'g': 'c',
		'h': 'd',
	}
	z_reflect_indices_transformation = np.array([
		8,       # 0  -> 8
		1,       # 1  -> 1
		5,       # 2  -> 5
		11,      # 3  -> 11
		4,       # 4  -> 4
		2,       # 5  -> 2
		10,      # 6  -> 10
		7,       # 7  -> 7
		0,       # 8  -> 0
		9,       # 9  -> 9
		6,       # 10 -> 6
		3,       # 11 -> 3
		-1,      # The last element (the -1 element, needs to map to -1)
	])


	# After performing a translation, some bitfields will remain unchanged or revert to ones that already exist,
	# that is because that bitfield is symmetric upon that combination of translations.
	# No worries, it just means we will override an existing with an equivalent form.

	# Create the resulting full marching cubes table and add the unique cases to it
	result = np.zeros((256, 5, 3), dtype=np.int32)
	result[unique_case_bitfield] = unique_case_table

	# Roll the unique cases once and add to the result
	roll1_bitfield = perform_bitfield_transformation(unique_case_bitfield, roll_bitfield_transformation)
	roll1_indices = perform_index_transformation(unique_case_table, roll_indices_transformation, reverse_normal=False) # Roll is the only transformation that does not effect the normals
	result[roll1_bitfield] = roll1_indices

	# Roll the previous result again and add to the result
	roll2_bitfield = perform_bitfield_transformation(roll1_bitfield, roll_bitfield_transformation)
	roll2_indices = perform_index_transformation(roll1_indices, roll_indices_transformation, reverse_normal=False)
	result[roll2_bitfield] = roll2_indices

	# Reflect the unique cases across the x axis
	x_reflect_bitfield = perform_bitfield_transformation(unique_case_bitfield, x_reflect_bitfield_transformation)
	x_reflect_indices = perform_index_transformation(unique_case_table, x_reflect_indices_transformation)
	result[x_reflect_bitfield] = x_reflect_indices

	# Reflect the unique cases across the y axis
	y_reflect_bitfield = perform_bitfield_transformation(unique_case_bitfield, y_reflect_bitfield_transformation)
	y_reflect_indices = perform_index_transformation(unique_case_table, y_reflect_indices_transformation)
	result[y_reflect_bitfield] = y_reflect_indices

	# Reflect the unique cases across the y axis
	z_reflect_bitfield = perform_bitfield_transformation(unique_case_bitfield, z_reflect_bitfield_transformation)
	z_reflect_indices = perform_index_transformation(unique_case_table, z_reflect_indices_transformation)
	result[z_reflect_bitfield] = z_reflect_indices


	# Reflect the rolled cases across the x axis
	roll_x_reflect_bitfield = perform_bitfield_transformation(roll1_bitfield, x_reflect_bitfield_transformation)
	roll_x_reflect_indices = perform_index_transformation(roll1_indices, x_reflect_indices_transformation)
	result[roll_x_reflect_bitfield] = roll_x_reflect_indices

	# Reflect the rolled cases across the y axis
	roll_y_reflect_bitfield = perform_bitfield_transformation(roll1_bitfield, y_reflect_bitfield_transformation)
	roll_y_reflect_indices = perform_index_transformation(roll1_indices, y_reflect_indices_transformation)
	result[roll_y_reflect_bitfield] = roll_y_reflect_indices

	# Reflect the rolled cases across the y axis
	roll_z_reflect_bitfield = perform_bitfield_transformation(roll1_bitfield, z_reflect_bitfield_transformation)
	roll_z_reflect_indices = perform_index_transformation(roll1_indices, z_reflect_indices_transformation)
	result[roll_z_reflect_bitfield] = roll_z_reflect_indices


	# Reflect the twice rolled cases across the x axis
	roll2_x_reflect_bitfield = perform_bitfield_transformation(roll2_bitfield, x_reflect_bitfield_transformation)
	roll2_x_reflect_indices = perform_index_transformation(roll2_indices, x_reflect_indices_transformation)
	result[roll2_x_reflect_bitfield] = roll2_x_reflect_indices

	# Reflect the twice rolled cases across the y axis
	roll2_y_reflect_bitfield = perform_bitfield_transformation(roll2_bitfield, y_reflect_bitfield_transformation)
	roll2_y_reflect_indices = perform_index_transformation(roll2_indices, y_reflect_indices_transformation)
	result[roll2_y_reflect_bitfield] = roll2_y_reflect_indices

	# Reflect the twice rolled cases across the y axis
	roll2_z_reflect_bitfield = perform_bitfield_transformation(roll2_bitfield, z_reflect_bitfield_transformation)
	roll2_z_reflect_indices = perform_index_transformation(roll2_indices, z_reflect_indices_transformation)
	result[roll2_z_reflect_bitfield] = roll2_z_reflect_indices



	# Reflect the x reflected results across the y axis
	xy_reflect_bitfield = perform_bitfield_transformation(x_reflect_bitfield, x_reflect_bitfield_transformation)
	xy_reflect_indices = perform_index_transformation(x_reflect_indices, x_reflect_indices_transformation)
	result[xy_reflect_bitfield] = xy_reflect_indices

	roll_xy_reflect_bitfield = perform_bitfield_transformation(roll_x_reflect_bitfield, x_reflect_bitfield_transformation)
	roll_xy_reflect_indices = perform_index_transformation(roll_x_reflect_indices, x_reflect_indices_transformation)
	result[roll_xy_reflect_bitfield] = roll_xy_reflect_indices

	roll2_xy_reflect_bitfield = perform_bitfield_transformation(roll2_x_reflect_bitfield, x_reflect_bitfield_transformation)
	roll2_xy_reflect_indices = perform_index_transformation(roll2_x_reflect_indices, x_reflect_indices_transformation)
	result[roll2_xy_reflect_bitfield] = roll2_xy_reflect_indices


	# Reflect the x reflected results across the y axis
	xy_reflect_bitfield = perform_bitfield_transformation(x_reflect_bitfield, y_reflect_bitfield_transformation)
	xy_reflect_indices = perform_index_transformation(x_reflect_indices, y_reflect_indices_transformation)
	result[xy_reflect_bitfield] = xy_reflect_indices

	roll_xy_reflect_bitfield = perform_bitfield_transformation(roll_x_reflect_bitfield, y_reflect_bitfield_transformation)
	roll_xy_reflect_indices = perform_index_transformation(roll_x_reflect_indices, y_reflect_indices_transformation)
	result[roll_xy_reflect_bitfield] = roll_xy_reflect_indices

	roll2_xy_reflect_bitfield = perform_bitfield_transformation(roll2_x_reflect_bitfield, y_reflect_bitfield_transformation)
	roll2_xy_reflect_indices = perform_index_transformation(roll2_x_reflect_indices, y_reflect_indices_transformation)
	result[roll2_xy_reflect_bitfield] = roll2_xy_reflect_indices


	# Reflect the z reflected results across the x axis
	zx_reflect_bitfield = perform_bitfield_transformation(z_reflect_bitfield, x_reflect_bitfield_transformation)
	zx_reflect_indices = perform_index_transformation(z_reflect_indices, x_reflect_indices_transformation)
	result[zx_reflect_bitfield] = zx_reflect_indices

	roll_zx_reflect_bitfield = perform_bitfield_transformation(roll_z_reflect_bitfield, x_reflect_bitfield_transformation)
	roll_zx_reflect_indices = perform_index_transformation(roll_z_reflect_indices, x_reflect_indices_transformation)
	result[roll_zx_reflect_bitfield] = roll_zx_reflect_indices

	roll2_zx_reflect_bitfield = perform_bitfield_transformation(roll2_z_reflect_bitfield, x_reflect_bitfield_transformation)
	roll2_zx_reflect_indices = perform_index_transformation(roll2_z_reflect_indices, x_reflect_indices_transformation)
	result[roll2_zx_reflect_bitfield] = roll2_zx_reflect_indices


	# Reflect the z reflected results across the y axis
	zy_reflect_bitfield = perform_bitfield_transformation(z_reflect_bitfield, y_reflect_bitfield_transformation)
	zy_reflect_indices = perform_index_transformation(z_reflect_indices, y_reflect_indices_transformation)
	result[zy_reflect_bitfield] = zy_reflect_indices

	roll_zy_reflect_bitfield = perform_bitfield_transformation(roll_z_reflect_bitfield, y_reflect_bitfield_transformation)
	roll_zy_reflect_indices = perform_index_transformation(roll_z_reflect_indices, y_reflect_indices_transformation)
	result[roll_zy_reflect_bitfield] = roll_zy_reflect_indices

	roll2_zy_reflect_bitfield = perform_bitfield_transformation(roll2_z_reflect_bitfield, y_reflect_bitfield_transformation)
	roll2_zy_reflect_indices = perform_index_transformation(roll2_z_reflect_indices, y_reflect_indices_transformation)
	result[roll2_zy_reflect_bitfield] = roll2_zy_reflect_indices


	# Reflect the zy reflected results across the x axis
	zyx_reflect_bitfield = perform_bitfield_transformation(zy_reflect_bitfield, x_reflect_bitfield_transformation)
	zyx_reflect_indices = perform_index_transformation(zy_reflect_indices, x_reflect_indices_transformation)
	result[zyx_reflect_bitfield] = zyx_reflect_indices

	roll_zyx_reflect_bitfield = perform_bitfield_transformation(roll_zy_reflect_bitfield, x_reflect_bitfield_transformation)
	roll_zyx_reflect_indices = perform_index_transformation(roll_zy_reflect_indices, x_reflect_indices_transformation)
	result[roll_zyx_reflect_bitfield] = roll_zyx_reflect_indices

	roll2_zyx_reflect_bitfield = perform_bitfield_transformation(roll2_zy_reflect_bitfield, x_reflect_bitfield_transformation)
	roll2_zyx_reflect_indices = perform_index_transformation(roll2_zy_reflect_indices, x_reflect_indices_transformation)
	result[roll2_zyx_reflect_bitfield] = roll2_zyx_reflect_indices


	# At this point, we now have all 256 marching cubes cases with indices 0 through 7 as described in the figure at the top of this function.
	# However, this does not give us a simple way to properly index vertex elements in the entire voxel grid.
	# The following array is a map between the 0 through 7 indices used in this function, and the appropriate strides needed to access that
	# vertex in a (a, b, c, 3) vertex matrix, where (a, b, c) is the shape of the voxel grid.

	#       +z                                           +z
	#        ^   +y                                       ^   +y
	#        |  /                                         |  /
	#        | /                                          | /
	#
	#        a-----0------b      -->  +x                v000---0000---v001      -->  +x
	#       /|           /|                              /|           /|
	#      2 |          6 |                           0002|       0012 |
	#     /  1         /  4                            /  0001      /  0011
	#    d-----3------c   |                  >>>    v100---1000---v101 |
	#    |   |        |   |                           |   |        |   |
	#    |   e-----8--|---f                           | v010--0100-|-v011
	#    7  /         9  /                          1001 /       1011  /
	#    | 5          | 10                            | 0102       | 0112
	#    |/           |/                              |/           |/
	#    h-----11-----g                             v110---1100---v111
	#                                                                               *v000 is a voxel at point (0, 0, 0)
	#                                                                               *0000 is a vertex at coordinates (0, 0, 0, 0)

	# Maps indices 0 through 7 to unravelled index offsets shown in the right diagram
	unravelled_offset_map = np.array([
		[0, 0, 0, 0],
		[0, 0, 0, 1],
		[0, 0, 0, 2],
		[1, 0, 0, 0],
		[0, 0, 1, 1],
		[0, 1, 0, 2],
		[0, 0, 1, 2],
		[1, 0, 0, 1],
		[0, 1, 0, 0],
		[1, 0, 1, 1],
		[0, 1, 1, 2],
		[1, 1, 0, 0],
		[0, 0, 0, 0], # This last guy is just a hack so that the ravelled offset map has a -1 as the last element
	])

	# Ravel the offsets so we can index a flattened vertex array
	# ravelled_offset_map = unravelled_offset_map[:, 0] * stride0 + unravelled_offset_map[:, 1] * stride1 + unravelled_offset_map[:, 2] * stride2 + unravelled_offset_map[:, 3] * stride3
	ravelled_offset_map = np.ravel_multi_index(unravelled_offset_map.T, (shape[0] + 2, shape[1] + 2, shape[2] + 2, shape[3])) # Use shape + 2 so we can pad the near and far planes of the voxel grid with 0
	ravelled_offset_map[-1] = -1

	# Now replace indices 0 through 7 with our vertex index offsets
	result_offsets = ravelled_offset_map[result] # Last element (-1st element) of ravelled_offset is also -1

	return result, result_offsets

def generate_voxel_bitfields(voxels):
	#       +z
	#        ^   +y
	#        |  /
	#        | /
	#
	#        a------------b      -->  +x
	#       /|           /|
	#      / |          / |
	#     /  |         /  |
	#    d------------c   |
	#    |   |        |   |
	#    |   e--------|---f
	#    |  /         |  /
	#    | /          | /
	#    |/           |/
	#    h------------g

	# Convert booleans a b c d e f h to a bitfield in this order  0b hfedcba  so that a is the least significant bit
	# The bitfield takes padding into consideration by not assigning values at the far planes and by physically padding
	# the near planes

	padded_voxels = np.zeros((voxels.shape[0] + 1, voxels.shape[1] + 1, voxels.shape[2] + 1), dtype=np.uint8) # Only need padded_voxels.shape + 1 because the far plane is excluded in the indexing, we only need physical padding for the near plane
	padded_voxels[1:, 1:, 1:] = voxels

	voxel_bitfields = np.zeros_like(padded_voxels)

	voxel_bitfields                 = padded_voxels.copy()                         # 0th bit (Is a True)
	voxel_bitfields[  :,   :, :-1] += np.uint8(padded_voxels[ :,  :, 1:] * 2)      # 1st bit (Is b True)
	voxel_bitfields[:-1,   :, :-1] += np.uint8(padded_voxels[1:,  :, 1:] * 4)      # 2nd bit (Is c True)
	voxel_bitfields[:-1,   :,   :] += np.uint8(padded_voxels[1:,  :,  :] * 8)      # 3rd bit (Is d True)
	voxel_bitfields[  :, :-1,   :] += np.uint8(padded_voxels[ :, 1:,  :] * 16)     # 4th bit (Is e True)
	voxel_bitfields[  :, :-1, :-1] += np.uint8(padded_voxels[ :, 1:, 1:] * 32)     # 5th bit (Is f True)
	voxel_bitfields[:-1, :-1, :-1] += np.uint8(padded_voxels[1:, 1:, 1:] * 64)     # 6th bit (Is g True)
	voxel_bitfields[:-1, :-1,   :] += np.uint8(padded_voxels[1:, 1:,  :] * 128)    # 7th bit (Is h True)

	return voxel_bitfields


def write_obj(vertex_indices, shape):
	print("Ignoring unused vertices")
	unique_indices = np.unique(vertex_indices)
	unravelled_unique_indices = np.unravel_index(unique_indices, (shape[0] + 2, shape[1] + 2, shape[2] + 2, shape[3])) # Unravel using shape + 2 to account for near and far plane padding

	print("Remapping used vertex indices")
	compacted_indices = np.arange(unique_indices.size)
	indices_map = {}
	indices_map.update(zip(unique_indices, compacted_indices))

	print("Creating obj")
	tris = vertex_indices.reshape(-1, 3) # Group by tris

	direction = unravelled_unique_indices[3]

	y_arr = -unravelled_unique_indices[0]*2 - (direction == 2)
	z_arr = -unravelled_unique_indices[1]*2 - (direction == 1)
	x_arr =  unravelled_unique_indices[2]*2 + (direction == 0)


	print(f' Vertices: {len(x_arr)}')
	print(f'Triangles: {len(tris)}')

	# vertices = np.stack([x_arr, y_arr, z_arr]).T
	# mesh = o3d.geometry.TriangleMesh(
	# 	o3d.utility.Vector3dVector(vertices),
	# 	o3d.utility.Vector3iVector(tris)
	# )

	# o3d.io.write_triangle_mesh("output.ply", mesh, compressed=True, write_vertex_colors=False, write_triangle_uvs=False, print_progress=True, write_ascii=False,)

	print()
	with open('output.obj', 'w') as f:
		f.write('# OBJ file\n')
		for i in range(len(unique_indices)):

			x = x_arr[i]
			y = y_arr[i]
			z = z_arr[i]

			# print(x, y, z)
			# print(unravelled_unique_indices[0][i], unravelled_unique_indices[1][i], unravelled_unique_indices[2][i], unravelled_unique_indices[3][i])
			# print(unique_indices[i])
			# print()

			f.write('v {0} {1} {2}\n'.format(x, y, z))

			if i % 50000 == 0:
				sys.stdout.write(f'Vertices: {i / len(unique_indices) * 100:0.2f}%                        \r')

		print()
		for i, tri in enumerate(tris):
			f.write('f')
			for vi in tri:
				f.write(' {0}'.format(indices_map[vi] + 1))
			f.write('\n')

			if i % 50000 == 0:
				sys.stdout.write(f'Triangles: {i / len(tris) * 100:0.2f}%                        \r')

def images_to_pcd(voxels):

	print()
	print('Generating point cloud')
	coords = np.where(voxels)
	xyz = np.zeros((coords[0].size, 3), dtype=np.int32)
	xyz[:, 1] = -coords[0]
	xyz[:, 2] = -coords[1]
	xyz[:, 0] =  coords[2]

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(xyz)
	o3d.io.write_point_cloud("output.ply", pcd, print_progress=True)

def pcd_to_voxels(file_path):
	pcd = o3d.io.read_point_cloud(file_path)
	points = np.asarray(pcd.points)

	# We want the coordinates as integers, but they might be screwed up after being edited.
	# So let's approximate a greatest common multiple of the distances between voxels by
	# just finding the minimum distance between any two voxels

	# First lets place the bounding box origin to the world origin
	min_x = points[:, 0].min()
	min_y = points[:, 1].max()
	min_z = points[:, 2].max()

	points[:, 0] -= min_x
	points[:, 1] -= min_y
	points[:, 2] -= min_z

	coords = np.zeros_like(points)
	coords[:, 0] = -points[:, 1]
	coords[:, 1] = -points[:, 2]
	coords[:, 2] =  points[:, 0] # We assume all coords are positive at this point

	# Now lets find the minimum spacing between points

	# First find spacing between all points
	diff_x = np.diff(coords[:, 0])
	diff_y = np.diff(coords[:, 1])
	diff_z = np.diff(coords[:, 2])

	# Now remove nonzero spacing
	diff_x = diff_x[diff_x > 0]
	diff_y = diff_y[diff_y > 0]
	diff_z = diff_z[diff_z > 0]

	# Now find minimum spacing in each axis
	min_diff_x = diff_x.min()
	min_diff_y = diff_y.min()
	min_diff_z = diff_z.min()

	min_diff = min(min_diff_x, min_diff_y, min_diff_z)

	# Now that we have the minimum distance approximation of a GCM,
	# let's divide all of our coordinate spacing by it
	coords //= min_diff

	coords = coords.astype(np.int32)
	i = coords[:, 0]
	j = coords[:, 1]
	k = coords[:, 2]

	shape = (i.max()+1, j.max()+1, k.max()+1)
	voxels = np.zeros(shape, dtype=bool)

	voxels[i, j, k] = True

	return voxels


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Converts a sequence of images into a voxelized obj')
	parser.add_argument('path', help='Path to folder of images or to a ply file')
	parser.add_argument('-e', '--export_for_edit', action='store_true', help="Don't process the images, just merge them into a ply file to be edited in another program")
	parser.add_argument('-r', '--radius', type=int, help='Radius of smoothing kernel', default=4)
	parser.add_argument('-s', '--sigma', type=float, help='sigma value of smoothing kernel', default=1.0)
	parser.add_argument('-t', '--threshold', type=float, help='threshold value for smoothing kerne', default=0.25)
	args = parser.parse_args()

	if os.path.isfile(args.path):
		voxels = pcd_to_voxels(args.path)
	else:
		voxels = images_to_voxels(args.path)

	if args.export_for_edit:

		images_to_pcd(voxels)

	else:
		print("\nSmoothing...")
		voxels = blur3d(voxels, radius=args.radius, sigma=args.sigma, threshold=args.threshold)

		print("Generating mesh")
		possible_vertices_shape = (voxels.shape[0], voxels.shape[1], voxels.shape[2], 3)

		debug_indices, marching_cubes_offsets = generate_marching_cubes_offsets(possible_vertices_shape)
		voxel_bitfields = generate_voxel_bitfields(voxels)

		offsets = marching_cubes_offsets[voxel_bitfields] # This is a (voxels.shape[0], voxels.shape[1], voxels.shape[2], 4, 3) matrix
		# debug = debug_indices[voxel_bitfields]

		padded_possible_vertices_shape = (voxels.shape[0] + 2, voxels.shape[1] + 2, voxels.shape[2] + 2) # Add 2 for near and far plane padding
		padded_possible_vertices_size = np.prod(padded_possible_vertices_shape)

		vertex_zero_index = np.arange(padded_possible_vertices_size).reshape(padded_possible_vertices_shape) * 3 # Index of 0th vertex for each voxel
		vertex_zero_index = vertex_zero_index[:-1, :-1, :-1] # Remove padding

		vertex_indices = vertex_zero_index[:, :, :, np.newaxis, np.newaxis] + offsets

		used_vertex_indices = vertex_indices[offsets != -1]
		write_obj(used_vertex_indices, possible_vertices_shape)

