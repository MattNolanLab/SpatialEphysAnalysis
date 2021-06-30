import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PostSorting.open_field_firing_fields

'''
calculates the border scores according to Solstad et al (2008)

"Putative border fields were identified first by identifying collections of neighboring pixels
 with firing rates higher than 0.3 times the maximum firing rate and covering a total area of 
 at least 200 cm2. For all experiments in square or rectangular environments, the coverage of
 a given wall of by a field was then estimated as the fraction of pixels along the wall that 
 was occupied by the field, and cM was defined as the maximum coverage of any single field 
 over any of the four walls of the environment. The mean firing distance dm was computed by 
 averaging the distance to the nearest wall over all pixels in the map belonging to some of its
 fields, weighted by the firing rate. To achieve this, the firing rate was normalized by its sum
 over all pixels belonging to some field, resembling a probability distribution. Finally, dm 
 was normalized by half of the shortest side of the environment (i.e. the largest possible 
 distance to its perimeter) so as to obtain a fraction between 0 and 1. A border score was defined 
 by comparing dm with the maximum coverage of any wall by a single field cM,   
 
 b = (cM - dm) / (cM + dm)
 
 Border scores ranged from -1 for cells with central firing fields to +1 for cells with fields that 
 perfectly line up along at least one entire wall. Intuitively, the border scores provide an idea of 
 the expansion of fields across walls rather than away from them. It should be noted that the measure 
 saturates when the width of the field approaches half the length of the environment.   
  
  ‘Border cells’ were defined as cells with border scores above 0.5. Only cells with stable border 
  fields (spatial correlation > 0.5) were included in the sample. In experiments with walls inserted 
  into the recording enclosure, the analysis was restricted to border cells with fields along a 
  single wall, i.e. cells where the border score for the preferred wall was at least twice as 
  high as the score for any of the remaining three walls."
  
Corner scores and cue scores are also formalised loosely following the b = (cM - dm) / (cM + dm) structure.
'''

def process_cue_data(spatial_firing, cue_location=0, open_field_size_cm=80, cue_size_cm=30):
    cue_scores = []
    threshold = 0.3

    for index, cluster in spatial_firing.iterrows():
        cluster_id = cluster.cluster_id

        firing_rate_map = cluster.firing_maps
        firing_rate_map = putative_border_fields_clip_by_firing_rate(firing_rate_map, threshold=threshold)

        '''
        fig, ax = plt.subplots()
        im = ax.imshow(firing_rate_map, cmap='jet')
        fig.tight_layout()
        plt.show()
        '''

        firing_fields_cluster, _ = get_firing_field_data(spatial_firing, index, threshold=threshold)
        firing_fields_cluster = fields2map(firing_fields_cluster, firing_rate_map)
        firing_fields_cluster = clip_fields_by_size(firing_fields_cluster, bin_size_cm=2.5)
        firing_fields_cluster = put_firing_rates_back(firing_fields_cluster, firing_rate_map)

        cue_score = calculate_cue_score(firing_fields_cluster, cue_location, open_field_size_cm, cue_size_cm, bin_size_cm=2.5)

        cue_scores.append(cue_score)

        #plot_fields_in_cluster_cue_scores(firing_fields_cluster, cue_score)

    spatial_firing['cue_score'] = cue_scores
    return spatial_firing


def process_border_data(spatial_firing):

    threshold = 0.3
    border_scores = []

    for index, cluster in spatial_firing.iterrows():
        cluster_id = cluster.cluster_id

        firing_rate_map = cluster.firing_maps.copy()
        firing_rate_map = putative_border_fields_clip_by_firing_rate(firing_rate_map, threshold=threshold)

        '''
        fig, ax = plt.subplots()
        im = ax.imshow(firing_rate_map, cmap='jet')
        fig.tight_layout()
        plt.show()
        '''

        firing_fields_cluster, _ = get_firing_field_data(spatial_firing, index, threshold=threshold)
        firing_fields_cluster = fields2map(firing_fields_cluster, firing_rate_map)
        firing_fields_cluster = clip_fields_by_size(firing_fields_cluster, bin_size_cm=2.5)
        firing_fields_cluster = put_firing_rates_back(firing_fields_cluster, firing_rate_map)

        border_score = calculate_border_score(firing_fields_cluster, bin_size_cm=2.5)

        border_scores.append(border_score)

        #plot_fields_in_cluster_border_scores(firing_fields_cluster, border_score)

    spatial_firing['border_score'] = border_scores
    return spatial_firing


def process_corner_data(spatial_firing):
    threshold = 0.3
    corner_scores = []

    for index, cluster in spatial_firing.iterrows():
        cluster_id = cluster.cluster_id

        firing_rate_map = cluster.firing_maps.copy()
        firing_rate_map = putative_border_fields_clip_by_firing_rate(firing_rate_map, threshold=threshold)

        '''
        fig, ax = plt.subplots()
        im = ax.imshow(firing_rate_map, cmap='jet')
        fig.tight_layout()
        plt.show()
        '''

        firing_fields_cluster, _ = get_firing_field_data(spatial_firing, index, threshold=threshold)
        firing_fields_cluster = fields2map(firing_fields_cluster, firing_rate_map)
        firing_fields_cluster = clip_fields_by_size(firing_fields_cluster, bin_size_cm=2.5)
        firing_fields_cluster = put_firing_rates_back(firing_fields_cluster, firing_rate_map)

        corner_score = calculate_corner_score(firing_fields_cluster, bin_size_cm=2.5, corner_param=0.2)

        corner_scores.append(corner_score)

        #plot_fields_in_cluster_corner_scores(firing_fields_cluster, corner_score)

    spatial_firing['corner_score'] = corner_scores
    return spatial_firing

def plot_fields_in_cluster_corner_scores(firing_fields_cluster, corner_score):
    for field in firing_fields_cluster:
        fig, ax = plt.subplots()
        im = ax.imshow(field, cmap='jet')
        fig.tight_layout()

        title = "corner_score: " + str(corner_score)
        ax.set_title(title)
        plt.show()

def plot_fields_in_cluster_cue_scores(firing_fields_cluster, cue_score):
    for field in firing_fields_cluster:
        fig, ax = plt.subplots()
        im = ax.imshow(field, cmap='jet')
        fig.tight_layout()

        title = "cue_score: " + str(cue_score)
        ax.set_title(title)
        plt.show()

def distance_matrix_corner(field, bin_size_cm):
    '''
        generates a matrix the same size as the rate map with elements
        corresponding to the mean shortest distance to the edge of the arena
        :param field: field rate map 2d np.array()
        :param bin_size_cm: int
        :return: distance matrix of same dimensions of field (unit cm)
        '''

    x, y = np.shape(field)

    corner_indices = [[0,0], [0,x-1], [y-1, 0], [y-1, x-1]]

    distance_matrix = np.zeros((x,y))

    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[0])):

            max_to_corner = []
            for corner in corner_indices:
                a = i-corner[1]
                b = j-corner[0]

                if np.round(a) == 0 and np.round(b) == 0:
                    c = 0
                else:
                    c = np.sqrt(((a*a)+(b*b)))

                max_to_corner.append(c)

            distance_matrix[i,j] = min(max_to_corner)

    distance_matrix = distance_matrix + 1
    distance_matrix = distance_matrix * bin_size_cm
    distance_matrix = distance_matrix - (bin_size_cm / 2)
    distance_matrix = distance_matrix/np.max(distance_matrix)

    return distance_matrix

def distance_matrix_cue(field, cue_location, open_field_size_cm, cue_size_cm, bin_size_cm):
    '''
    generates a matrix the same size as the rate map with elements
        corresponding to the mean shortest distance to the cue
    :param field: field rate map 2d np.array()
    :param bin_size_cm: int
    :param cue_location: this indicates which side of the arena has the cue on it either 0 = left side of rate map, 1 = top, 2 = right or 3 = bottom
    :param open_field_size_cm: tuple dimesions of open field (x, y) in cm
    :param cue_size_cm: length of cue, presumed to be at a central location along one of the sides specified in cue location param
    :return: distance matrix of same dimensions of field (unit cm)
    '''

    x, y = np.shape(field)

    distance_matrix = np.zeros((x, y))

    cue_indices = get_cue_indices(field, cue_location, open_field_size_cm, cue_size_cm, bin_size_cm=2.5)

    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[0])):

            max_to_cue = []
            for cue_index in cue_indices:
                a = i-cue_index[0]
                b = j-cue_index[1]

                if np.round(a) == 0 and np.round(b) == 0:
                    c = 0
                else:
                    c = np.sqrt(((a*a)+(b*b)))

                max_to_cue.append(c)

            distance_matrix[i,j] = min(max_to_cue)

    distance_matrix = distance_matrix + 1
    distance_matrix = distance_matrix * bin_size_cm
    distance_matrix = distance_matrix - (bin_size_cm / 2)
    distance_matrix = distance_matrix/np.max(distance_matrix)

    '''
    fig, ax = plt.subplots()
    im = ax.imshow(distance_matrix, cmap='jet')
    fig.tight_layout()
    plt.show()
    '''

    return distance_matrix

def get_cue_indices(field, cue_location, open_field_size_cm, cue_size_cm, bin_size_cm=2.5):

    cue_coverage = float(cue_size_cm) / float(open_field_size_cm)

    x, y = np.shape(field)

    distance_matrix = np.zeros((x, y))

    if cue_location == 0:
        side_indices = [(i, j) for i in range(x) for j in [0]]
    elif cue_location == 1:
        side_indices = [(i, j) for i in [0] for j in range(y)]
    elif cue_location == 2:
        side_indices = [(i, j) for i in range(x) for j in [y - 1]]
    elif cue_location == 3:
        side_indices = [(i, j) for i in [x - 1] for j in range(y)]

    if cue_coverage < 1:
        cue_indices = side_indices[int(int(np.floor((len(side_indices)) / 2)) - (len(side_indices) * cue_coverage / 2)):
                                   int(int(np.ceil((len(side_indices)) / 2)) + (
                                               len(side_indices) * cue_coverage / 2)) + 1]
    else:
        cue_indices = side_indices

    return cue_indices

def length_of_cue_wall(field, cue_location):

    if cue_location == 0:
        return len(field[:,0])
    elif cue_location == 1:
        return len(field[0])
    elif cue_location == 2:
        return len(field[:, 0])
    elif cue_location == 3:
        return len(field[0])

def calculate_cue_score(firing_fields_cluster, cue_location, open_field_size_cm, cue_size_cm, bin_size_cm=2.5):
    '''
    :param firing_fields_cluster:
    :param bin_size_cm:
    :param cue_location: this indicates which side of the arena has the cue on it either 0 = left side of rate map, 1 = top, 2 = right or 3 = bottom
    # TODO: change this to north, east, west and south, all relative to access side of open field
    :param open_field_size_cm: tuple dimesions of open field (x, y) in cm
    :param cue_size_cm: length of cue, presumed to be at a central location along one of the sides specified in cue location param
    :return: cue score
    '''


    # only execute if there are firing fields to analyse
    if len(firing_fields_cluster) > 0:

        normalised_distance_mat = distance_matrix_cue(firing_fields_cluster[0], cue_location, open_field_size_cm, cue_size_cm, bin_size_cm)
        cue_indices = get_cue_indices(firing_fields_cluster[0], cue_location, open_field_size_cm, cue_size_cm, bin_size_cm)

        dm = []  # takes on new meaning of mean firing distance to cue

        maxcM = 0


        for field in firing_fields_cluster:

            field_on_cue_count = 0

            field_count = field.copy()
            field_count[field_count > 0] = 1

            for cue_index in cue_indices:
                if field_count[cue_index]==1:
                    field_on_cue_count += 1

            cM = field_on_cue_count/length_of_cue_wall(firing_fields_cluster[0], cue_location)

            if cM > maxcM:
                maxcM = cM

            normalized_field = field / np.sum(field)

            dm_for_field = np.multiply(normalized_field,
                                       normalised_distance_mat)  # weight by shortest distance to the perimeter
            dm_for_field = np.sum(dm_for_field)

            dm.append(dm_for_field)

        dm_all_fields = np.mean(dm)

        # final measure of mean firing distance
        dm = dm_all_fields.copy()
        final_cM = maxcM

        cue_score = (final_cM - dm) / (final_cM + dm)

        return cue_score

def calculate_corner_score(firing_fields_cluster, bin_size_cm, corner_param):
    '''

    :param firing_fields_cluster:
    :param bin_size_cm:
    :param corner_param: defines the proportion of the wall that is used to count the percentage coverage about a corner > 0 and < 0.5.
    note: this parameter is used as unlike cue and bordeer scores, the corner is defined by a single point (or bin) and thus needs a more dilute measure to accommodate fields
    that span multiple bins
    TODO: Discuss best way to get around this
    :return:
    '''

    # only execute if there are firing fields to analyse
    if len(firing_fields_cluster)>0:

        normalised_distance_mat = distance_matrix_corner(firing_fields_cluster[0], bin_size_cm)


        dm = [] # takes on new meaning of mean firing distance to corner

        maxcM = 0

        for field in firing_fields_cluster:

            field_count = field.copy()
            field_count[field_count > 0] = 1

            corner_bins_wall1 = int(np.ceil(corner_param * len(field_count[0])))
            corner_bins_wall2 = int(np.ceil(corner_param * len(field_count[:,0])))
            corner_bins_wall3 = int(np.ceil(corner_param * len(field_count[-1])))
            corner_bins_wall4 = int(np.ceil(corner_param * len(field_count[:,-1])))

            corner1_cM = np.sum(field_count[0, 0:corner_bins_wall1]) + np.sum(field_count[:,0][0:corner_bins_wall2])
            if field_count[0,0] == 1:
                corner1_cM -= 1

            corner2_cM = np.sum(field_count[:,0][-corner_bins_wall2:]) + np.sum(field_count[-1, 0:corner_bins_wall3])
            if field_count[-1, 0] == 1:
                corner2_cM -= 1

            corner3_cM = np.sum(field_count[-1, -corner_bins_wall3:]) + np.sum(field_count[:,-1][-corner_bins_wall4:])
            if field_count[-1, -1] == 1:
                corner3_cM -= 1

            corner4_cM = np.sum(field_count[:,-1][0:corner_bins_wall4]) + np.sum(field_count[0, -corner_bins_wall1:])
            if field_count[0, -1] == 1:
                corner4_cM -= 1

            corner1_cM = corner1_cM / (corner_bins_wall1 + corner_bins_wall2 - 1)
            corner2_cM = corner2_cM / (corner_bins_wall2 + corner_bins_wall3 - 1)
            corner3_cM = corner3_cM / (corner_bins_wall3 + corner_bins_wall4 - 1)
            corner4_cM = corner4_cM / (corner_bins_wall4 + corner_bins_wall1 - 1)

            # reassign max cM if found bigger in a different field or wall
            if corner1_cM>maxcM:
                maxcM= corner1_cM
            elif corner2_cM>maxcM:
                maxcM= corner2_cM
            elif corner3_cM > maxcM:
                maxcM = corner3_cM
            elif corner4_cM > maxcM:
                maxcM = corner4_cM

            normalized_field = field/np.sum(field)

            dm_for_field = np.multiply(normalized_field, normalised_distance_mat)  # weight by shortest distance to the perimeter
            dm_for_field = np.sum(dm_for_field)

            dm.append(dm_for_field)

        dm_all_fields = np.mean(dm)

        # final measure of mean firing distance
        dm = dm_all_fields.copy()
        cM = maxcM

        corner_score = (cM - dm) / (cM + dm)

        return corner_score


def put_firing_rates_back(firing_fields_cluster, firing_rate_map):

    new = []
    for field in firing_fields_cluster:
        new.append(np.multiply(field, firing_rate_map))

    return new

def calculate_border_score(firing_fields_cluster, bin_size_cm):

    # only execute if there are firing fields to analyse
    if len(firing_fields_cluster)>0:

        normalised_distance_mat = distance_matrix_border(firing_fields_cluster[0], bin_size_cm)

        dm = []

        maxcM = 0

        for field in firing_fields_cluster:

            field_count = field.copy()
            field_count[field_count > 0] = 1

            wall1_cM = np.sum(field_count[0])/len(field_count[0])
            wall2_cM = np.sum(field_count[:,0])/len(field_count[:,0])
            wall3_cM = np.sum(field_count[:,-1])/len(field_count[:,-1])
            wall4_cM = np.sum(field_count[-1])/len(field_count[-1])

            # reassign max cM if found bigger in a different field or wall

            if wall1_cM>maxcM:
                maxcM= wall1_cM
            elif wall2_cM>maxcM:
                maxcM= wall2_cM
            elif wall3_cM > maxcM:
                maxcM = wall3_cM
            elif wall4_cM > maxcM:
                maxcM = wall4_cM

            normalized_field = field/np.sum(field)

            dm_for_field = np.multiply(normalized_field, normalised_distance_mat)  # weight by shortest distance to the perimeter
            dm_for_field = np.sum(dm_for_field)

            dm.append(dm_for_field)

        dm_all_fields = np.mean(dm)

        # final measure of mean firing distance
        dm = dm_all_fields.copy()
        cM = maxcM

        border_score = (cM - dm) / (cM + dm)

        return border_score

    else:
        border_score = np.nan

        # if no fields are found return NaN for border score (discredit these)
        return border_score


def distance_matrix_border(field, bin_size_cm):
    '''
    generates a matrix the same size as the rate map with elements
    corresponding to the mean shortest distance to the edge of the arena
    :param field: field rate map 2d np.array()
    :param bin_size_cm: int
    :return: distance matrix of same dimensions of field (unit cm)
    '''

    x, y = np.shape(field)

    r = np.arange(x)
    r2 = np.arange(y)

    d1 = np.minimum(r, r[::-1])
    d2 = np.minimum(r2, r2[::-1])

    distance_matrix = np.minimum.outer(d1, d2)

    distance_matrix = distance_matrix + 1
    distance_matrix = distance_matrix * bin_size_cm
    distance_matrix = distance_matrix - (bin_size_cm/2)

    distance_matrix = distance_matrix/np.max(distance_matrix) # normalise to largest distance to border

    return distance_matrix


def stack_fields(firing_fields_clusters):
    '''
    this functions stacks the firing fields back together
    :param firing_fields_clusters: masked rate maps with individual firing fields
    :return: rate map with all firing fields 0= out of field, 1 = in field
    '''
    stacked = np.sum(firing_fields_clusters, axis=0)
    return stacked


def plot_fields_in_cluster(firing_fields_cluster):
    for field in firing_fields_cluster:
        fig, ax = plt.subplots()
        im = ax.imshow(field, cmap='jet')
        plt.show()

def plot_fields_in_cluster_border_scores(firing_fields_cluster, border_score):
    for field in firing_fields_cluster:
        fig, ax = plt.subplots()
        im = ax.imshow(field, cmap='jet')
        fig.tight_layout()

        title = "border_score: " + str(border_score)
        ax.set_title(title)
        plt.show()

def fields2map(firing_fields_cluster, firing_rate_map_template):
    '''
    :param firing_field_cluster: coordinates of firing fields for a given cluster
    :param firing_rate_map_template: example firing rate map to copy structure
    :return: rate map per field
    '''
    firing_fields = []

    for field in firing_fields_cluster:
        field_firing = firing_rate_map_template.copy() * 0

        for i in range(len(field)):
            field_firing[field[i][0]][field[i][1]] = 1

        firing_fields.append(field_firing)

    return firing_fields

def clip_fields_by_size(masked_rate_maps, bin_size_cm=2.5):
    '''
    clips the fields in the firing rate map if the neighbouring regions don't sum to 200cm2
    :param firing_rate_map: smoothened firing rate map, preclipped by max firing rate
    :return: clipped firing rate
    '''
    bin_volume_cm2 = bin_size_cm*bin_size_cm

    new_masked_rate_maps = []

    for field in masked_rate_maps:
        if np.sum(field*bin_volume_cm2)>200:   # as specified by Solstad et al (2008), only fields larger than 200cm2 are considered
            new_masked_rate_maps.append(field)

    return new_masked_rate_maps



def putative_border_fields_clip_by_firing_rate(firing_rate_map, threshold):
    '''
    clips the fields in the firing rate map if the firing rate is below 0.3x max firing rate
    :param firing_rate_map: smoothened firing rate map
    :return: firing_rate_map clipped by 0.3x max firing rate
    '''

    max_firing = np.max(firing_rate_map)
    firing_rate_map[firing_rate_map < threshold*max_firing] = 0
    return firing_rate_map


'''
Functions below taken from open_field_firing_fields,
modified to fit the purposes of border field detection
'''


# return indices of neighbors of bin considering borders
def find_neighbors(bin_to_test, max_x, max_y):
    x = bin_to_test[0]
    y = bin_to_test[1]

    neighbors = [[x, y+1], [x, y-1], [x+1, y], [x-1, y]]

    if x == max_x:
        neighbors = [[x, y+1], [x, y-1], [x-1, y]]
    if y == max_y:
        neighbors = [[x, y-1], [x+1, y], [x-1, y]]
    if x == max_x and y == max_y:
        neighbors = [[x, y-1], [x-1, y]]
    if x == 0:
        neighbors = [[x, y+1], [x, y-1], [x+1, y]]
    if y == 0:
        neighbors = [[x, y+1], [x+1, y], [x-1, y]]
    if x == 0 and y == 0:
        neighbors = [[x, y+1], [x+1, y]]

    if x == max_x and y == 0:
        neighbors = [[x, y+1], [x-1, y]]

    if y == max_y and x == 0:
        neighbors = [[x, y-1], [x+1, y]]

    return neighbors


# return the masked rate map and change the neighbor's indices to 1 if they are above threshold
def find_neighborhood(masked_rate_map, rate_map, firing_rate_of_max, threshold):
    changed = False
    threshold = firing_rate_of_max * threshold / 100

    firing_field_bins = np.array(np.where(masked_rate_map > 0))
    firing_field_bins = firing_field_bins.T

    for bin_to_test in firing_field_bins:
        masked_rate_map[bin_to_test[0], bin_to_test[1]] = 2
        neighbors = find_neighbors(bin_to_test, max_x=(masked_rate_map.shape[0]-1), max_y=(masked_rate_map.shape[1]-1))
        for neighbor in neighbors:
            if masked_rate_map[neighbor[0], neighbor[1]] == 2:
                continue

            firing_rate = rate_map[neighbor[0], neighbor[1]]
            if firing_rate >= threshold:
                masked_rate_map[neighbor[0], neighbor[1]] = 1
                changed = True

    return masked_rate_map, changed


# check if the detected field is big enough to be a firing field
def test_if_field_is_big_enough(field_indices, max_pixels_in_field = 32):
    number_of_pixels = len(field_indices)
    if number_of_pixels > max_pixels_in_field:
        return True
    return False


# this is to avoid identifying the whole rate map as a field
def test_if_field_is_small_enough(field_indices, rate_map):
    number_of_pixels_in_field = len(field_indices)
    number_of_pixels_on_map = len(rate_map.flatten())
    if number_of_pixels_in_field > 3*(number_of_pixels_on_map/4):    # if field is bigger than 3/4 of map, reject field
        return False
    else:
        return True


def get_field_edge_values(field_indices):
    x_min = np.array(field_indices)[:, 0].min()
    x_max = np.array(field_indices)[:, 0].max()
    y_min = np.array(field_indices)[:, 1].min()
    y_max = np.array(field_indices)[:, 1].max()
    return x_min, x_max, y_min, y_max

'''
def test_if_field_is_not_too_spread_out(field_indices, rate_map):
    x_min, x_max, y_min, y_max = get_field_edge_values(field_indices)
    if (x_max - x_min) >= len(rate_map) / 2:
        return False
    if (y_max - y_min) >= len(rate_map) / 2:
        return False
    return True
'''


def ensure_the_field_does_not_have_a_hole_in_the_middle(field_indices):
    x_min, x_max, y_min, y_max = get_field_edge_values(field_indices)
    middle_x = x_max - int((x_max - x_min) / 2)
    middle_y = y_max - int((y_max - y_min) / 2)
    if [middle_x, middle_y] not in field_indices.tolist():
        return False
    return True


# test if the firing rate of the detected local maximum is higher than average + std firing
def test_if_highest_bin_is_high_enough(rate_map, highest_rate_bin):
    flat_rate_map = rate_map.flatten()
    rate_map_without_removed_fields = np.take(flat_rate_map, np.where(flat_rate_map >= 0))
    average_rate = np.mean(rate_map_without_removed_fields)
    std_rate = np.std(rate_map)

    firing_rate_of_highest_bin = rate_map[highest_rate_bin[0], highest_rate_bin[1]]
    if firing_rate_of_highest_bin < 0.1:
        return False

    if firing_rate_of_highest_bin > average_rate + std_rate:
        return True
    else:
        return False


# find indices for an individual firing field
def find_current_maxima_indices(rate_map, threshold):
    highest_rate_bin = np.unravel_index(rate_map.argmax(), rate_map.shape)
    found_new = test_if_highest_bin_is_high_enough(rate_map, highest_rate_bin)
    max_fr = rate_map[highest_rate_bin]
    if found_new is False:
        return None, found_new, None
    #plt.imshow(rate_map)
    # plt.scatter(highest_rate_bin[1], highest_rate_bin[0], marker='o', s=500, color='yellow')
    masked_rate_map = np.full((rate_map.shape[0], rate_map.shape[1]), 0)
    masked_rate_map[highest_rate_bin] = 1
    changed = True
    while changed:
        masked_rate_map, changed = find_neighborhood(masked_rate_map, rate_map, rate_map[highest_rate_bin], threshold=threshold)

    field_indices = np.array(np.where(masked_rate_map > 0)).T
    found_new = test_if_field_is_big_enough(field_indices)
    if found_new is False:
        return None, found_new, None

    #found_new = test_if_field_is_small_enough(field_indices, rate_map)
    #if found_new is False:
    #    return None, found_new, None

    found_new = ensure_the_field_does_not_have_a_hole_in_the_middle(field_indices)
    if found_new is False:
        return None, found_new, None
    return field_indices, found_new, max_fr


# mark indices of firing fields that are already found (so we don't find them again)
def remove_indices_from_rate_map(rate_map, indices):
    for index in indices:
        rate_map[index[0], index[1]] = -10
    return rate_map


# find firing fields and maximum firing rates for each field for a cluster
def get_firing_field_data(spatial_firing, cluster, threshold):
    firing_fields_cluster = []
    max_firing_rates_cluster = []
    rate_map = spatial_firing.firing_maps[cluster].copy()
    found_new = True
    while found_new:
        field_indices, found_new, max_firing_rate = find_current_maxima_indices(rate_map, threshold=threshold)
        if found_new:
            firing_fields_cluster.append(field_indices)
            max_firing_rates_cluster.append(max_firing_rate)
            rate_map = remove_indices_from_rate_map(rate_map, field_indices)
    return firing_fields_cluster, max_firing_rates_cluster


