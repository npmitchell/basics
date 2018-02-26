import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys

"""Low level module with functions for analyzing linesegments, linesegment intersections, line intersections, and 
getting closest points on a linesegment.
Note that this module shall not import any other lepm module.
Most useful functions are linesegs_intersect_linesegs() and mindist_from_multiple_linesegs()

Note that ILPM.path has cl_src code with a GPU-accelerated get_crossings function, which takes the projection of 3d
data and computes the intersections. The analogous functions included in THIS module is quasi-optimized for cases where
there are many linesegments which are not near each other, and therefore one doesn't need to compute the
intersections of their lines (the lines defined by the linesegs) at all. However, this code is NOT GPU accelerated.
"""


# def points_into_linesegs(pts_a, pts_b):
#     """Turn two Nx2 arrays of xy points into a single Nx4 array, to be used as endpoints denoting linesegs"""
#     return np.hstack((pts_a, pts_b))


def perp(a):
    """Get vector perpendicular to input vector a

    Parameters
    ----------
    a : 1 x 2 float array
        The input vector

    Returns
    -------
    b : 1 x 2 float array
        The output vector perpendicular to a
    """
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def perp_vects(a):
    """Get vector perpendicular to input vectors a

    Parameters
    ----------
    a : N x 2 float array
        The input vector

    Returns
    -------
    b : N x 2 float array
        The output vectors perpendicular to a
    """
    b = np.empty_like(a)
    b[:, 0] = -a[:, 1]
    b[:, 1] = a[:, 0]
    return b


def bounding_boxes_intersect(linesegs_a, linesegs_b):
    """For linesegments a and b, look to see if their bounding boxes overlap, returning an N x M boolean array if a is
    N x 4 and b is M x 4. Each linesegment is denoted by [x1, y1, x2, y2] connecting (x1, y1) to (x2, y2).
    """
    aa = linesegs_a
    bb = linesegs_b

    # get centers of each lineseg as the center of a bounding box
    xcols = [0, 2]
    ycols = [1, 3]
    # print 'aa = ', aa
    # print 'bb = ', bb
    # print 'np.shape(aa[:, xcols]) = ', np.shape(aa[:, xcols])
    # print 'np.shape(aa[:, ycols]) = ', np.shape(aa[:, ycols])
    # print 'bb[:, xcols]= ', bb[:, xcols]

    acent = np.dstack((np.mean(aa[:, xcols], axis=1), np.mean(aa[:, ycols], axis=1)))[0]
    bcent = np.dstack((np.mean(bb[:, xcols], axis=1), np.mean(bb[:, ycols], axis=1)))[0]

    # get widths and heights of each bounding box
    aw = np.abs(aa[:, xcols[0]] - aa[:, xcols[1]])
    bw = np.abs(bb[:, xcols[0]] - bb[:, xcols[1]])
    ah = np.abs(aa[:, ycols[0]] - aa[:, ycols[1]])
    bh = np.abs(bb[:, ycols[0]] - bb[:, ycols[1]])

    bbintersect = np.array([np.logical_and(np.abs(acent[ii, 0] - bcent[:, 0]) * 2 < (aw[ii] + bw),
                                           np.abs(acent[ii, 1] - bcent[:, 1]) * 2 < (ah[ii] + bh))
                            for ii in range(len(acent))])
    return bbintersect


def intersection_lines(a1, a2, b1, b2):
    """Find line intersection using two points on each line
    see Computer Graphics by F.S. Hill

    Parameters
    ----------
    a1 : 1 x 2 float array
        endpoint 1 for the first lineseg defining the first line
    a2 : 1 x 2 float array
        endpoint 2 for the first lineseg defining the first line
    b1 : 1 x 2 float array
        endpoint 1 for the second lineseg defining the second line
    b2 : 1 x 2 float array
        endpoint 2 for the second lineseg defining the second line

    Returns
    ----------
    intersect : 1 x 2 float array
        The intersection of the two lines defined by pts (a1,a2) and (b1,b2)
    """
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    intersect = (num / denom.astype(float)) * db + b1
    return intersect


def intersections_many_lines_2d(lines):
    """Find the intersection points of all pairs of intersecting lines, with a dict giving which lines are connected to
    which

    Parameters
    ----------
    lines : N x 4 float array
        Each row is a line defined by two points living on the line: lines[0] = np.array([x0, y0, x1, y1])

    Returns
    -------
    intrx : dict
        for each line intersection (specified by the int indices of the intersecting lines as a key), give pt where
        the intersection exists (2 x 1 float array val)
    """
    intrx = {}
    ind = 0
    for line in lines:
        a1 = np.array([line[0], line[1]])
        a2 = np.array([line[2], line[3]])
        # Note here that we avoid repeating intersection computations by ignoring all previously examined lines
        for jj in range(ind + 1, len(lines)):
            b1 = np.array([lines[jj][0], lines[jj][1]])
            b2 = np.array([lines[jj][2], lines[jj][3]])
            inx = intersection_lines(a1, a2, b1, b2)
            if inx is not None:
                intrx[(ind, jj)] = inx

        # next time around, ignore previous lines
        ind += 1

    return intrx


def lines_intersect_which_lines_2d(lines):
    """Find the intersection points of all pairs of intersecting lines, with a dict giving which lines are connected to
    which

    Parameters
    ----------
    lines : N x 4 float array
        Each row is a line defined by two points living on the line: lines[0] = np.array([x0, y0, x1, y1])

    Returns
    -------
    intrx : dict
        for each line (specified by the int index as a key), give list of which lines it does intersect
        (int val[0][i]), and where is the intersection (2 x 1 float array val[1][i])
    """
    intrxs = {}
    ind = 0
    for line in lines:
        a1 = np.array([line[0], line[1]])
        a2 = np.array([line[2], line[3]])
        # Note here that we avoid repeating intersection computations by ignoring all previously examined lines
        for jj in range(ind + 1, len(lines)):
            b1 = np.array([lines[jj][0], lines[jj][1]])
            b2 = np.array([lines[jj][2], lines[jj][3]])
            inx = intersection_lines(a1, a2, b1, b2)
            if inx is not None:
                if ind not in intrxs:
                    intrxs[ind] = ([jj], [inx])
                else:
                    intrxs[ind][0].append(jj)
                    intrxs[ind][1].append(inx)
                if jj not in intrxs:
                    intrxs[jj] = ([ind], [inx])
                else:
                    intrxs[jj][0].append(ind)
                    intrxs[jj][1].append(inx)

        # next time around, ignore previous lines
        ind += 1

    return intrxs


def min_or_NaN(arr):
    """Extend the np.min() function to allow empty arrays or other Error-raising objects, in which case return a nan"""
    try:
        return np.min(arr)
    except:
        return np.NaN


def nanargmin_or_NaN(arr):
    """Extend the np.nanargmin() function to allow empty arrays or other Error-raising objects, in which case return
    a nan"""
    try:
        return np.nanargmin(arr)
    except:
        return np.NaN


def max_or_NaN(arr):
    """Extend the np.max() function to allow empty arrays or other Error-raising objects, in which case return a nan"""
    try:
        return np.max(arr)
    except:
        return np.NaN


def nanargmax_or_NaN(arr):
    """Extend the np.nanargmax() function to allow empty arrays or other Error-raising objects, in which case return
    a nan"""
    try:
        return np.nanargmax(arr)
    except:
        return np.NaN


def network_from_intersections(lines):
    """From a collection of lines defined by two points on each line, create a network of vtcs and bonds connecting
    their intersections"""
    from lepm.data_handling import dist_pts_along_vec
    intx = intersections_many_lines_2d(lines)
    # intx is returned as a dict, with keys being the index of each line
    # Plan: for each intersection, store which lines the intersection uses.
    # To form bonds, candidate NN will share at least one line. In fact, they will be the nearest vtcs on a given line
    # vtcs = []
    # bonds = []
    # for lineii in intx:
    #     # Add all intersections on this line to the vtcs list, even if it has already been counted
    #     new_vtcs = intx[lineii][1]
    #     vtcs.append(new_vtcs)
    #
    #     # For each vertex, form a bond between this vertex and the other nearest vertices on either side of the point
    #     # along this line --> pdists are projected distances
    #     vec = np.array([lineii[0:2], lineii[2:4]])
    #     pdists = dh.dist_pts_along_vec(new_vtcs, new_vtcs, vec)
    #     # For each vertex, look which vects have the shortest distance along positive direction of lineii's defining
    #     # vector and which are negative (up_neighbor and down_neighbor)
    #     vtxii = len(vtcs) - len(new_vtcs)
    #     for row in pdists:
    #         posinds = np.where(row > 0)[0]
    #         # upnbr is the row index of the neighboring intersection nearest to the considered intersection
    #         neginds = np.where(row < 0)[0]
    #         try:
    #             upnbr = posinds[np.argmin(posinds)]
    #             bonds.append([vtxii, vtxii + upnbr])
    #         except ValueError:
    #             upnbr = None
    #         try:
    #             downnbr = neginds[np.argmin(neginds)]
    #             bonds.append([vtxii, vtxii + downnbr])
    #         except ValueError:
    #             downnbr = None
    #
    #         vtxii += 1
    #
    # # Could do:
    # # Add each intersection on this line to the vtcs list that has not already been counted
    # # but didn't.
    #
    # # Now remove the vertices which were double counted
    # vtcs_trimmed, order, ui = dh.args_unique_rows_threshold(vtcs, thres=1e-12)
    # from lepm.lattice_elasticity import remove_pts
    # remove_pts()

    # Unpack all the vertices of the network we are forming, but also make an array from the pairs of lines for lookup
    # This forms xy.
    vtcs = np.zeros((len(intx), 2), dtype=float)
    pairs = np.zeros((len(intx), 2), dtype=int)
    ii = 0
    for pair in intx:
        # Add coordinate of intersection
        vtcs[ii] = intx[pair]
        pairs[ii] = np.array([pair[0], pair[1]])
        ii += 1

    # Now form the bond list
    bonds = []
    for lineii in lines:
        # For the lower index line, form a bond between the current intersection and the other nearest vertices on
        # either side of the point along this line --> pdists are projected distances
        vec = np.array([lineii[0:2], lineii[2:4]])
        # Grab all vertices (intersections) on this line
        linevxii = np.where(pairs == lineii)[0]
        # Grab their locations too
        linevtcs = vtcs[linevxii]

        # Now consider each vertex on the line, but do it in a vectorized way
        # Form vectors from current vtx to other vtcs on this line (linevtcs)
        print 'line_segments.network_from_intersections(): linevxii = ', pairs == lineii
        print 'line_segments.network_from_intersections(): linevxii = ', linevxii
        print 'line_segments.network_from_intersections(): linevtcs = ', linevtcs
        pdists = dist_pts_along_vec(linevtcs, linevtcs, vec)
        # For each vertex, look which vects have the shortest distance along positive direction of lineii's defining
        # vector and which are negative (up_neighbor and down_neighbor)
        ind = 0
        for row in pdists:
            vtxii = linevxii[ind]

            posinds = np.where(row > 0)[0]
            # upnbr is the row index of the neighboring intersection nearest to the considered intersection
            neginds = np.where(row < 0)[0]
            try:
                upnbr = posinds[np.argmin(posinds)]
                bonds.append([vtxii, linevxii[upnbr]])
            except ValueError:
                '''There is no vertex upstream of this one along the current line'''
                upnbr = None

            try:
                downnbr = neginds[np.argmin(neginds)]
                bonds.append([vtxii, linevxii[downnbr]])
            except ValueError:
                '''There is no vertex downstream of this one along the current line'''
                downnbr = None

            ind += 1

    BL = np.array(bonds)
    return vtcs, BL


def intersection_linesegs(a1, a2, b1, b2, thres=1e-6):
    """Find line segment intersection of a single linesegment (a1, a2) with multiple other linesegs (b1,b2),
    using pairwise comparison of endpts of lineseg (a1, a2) with all linesegs specified by (b1, b2) pairs.
    b1 and b2 must be the same shape (they are paired endpoints), but need not be 1 x 2--> ie can be N x 2 arrays.
    If size(b1) = 2, finds intersections of (a1, a2) and (b1, b2) and returns True if they intersect.
    If size(b1) > 2, finds intersections of (a1, a2) with (b1[ii], b2[ii]) for each ii in range(len(b1)) and returns
    boolean array of whether the lineseg (a1, a2) intersects each lineseg (b1, b2).
    see Computer Graphics by F.S. Hill
    http://www.cs.mun.ca/~rod/2500/notes/numpy-arrays/numpy-arrays.html
    http://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    http://geomalgorithms.com/a05-_intersect-1.html

    Parameters
    ----------
    a1 : 1 x 2 float array
        endpoint 1 for the first lineseg
    a2 : 1 x 2 float array
        endpoint 2 for the first lineseg
    b1 : N x 2 float array
        endpoint 1 for the second lineseg
    b2 : N x 2 float array
        endpoint 2 for the second lineseg
    thres : float
        minimum distance for admitting intersection

    Returns
    ----------
    intersect : N x 2 float array or None
        If N =1, returns the intersection of the two linesegments (a1,a2) and (b1,b2) or None if no intersection
        If N >1, returns the intersections
    does_intersect: bool (if N=1) or N x 1 bool array (if N>1)
        Whether the linesegs in a
    """
    # Check if this is a single pair of linesegments possibly intersecting or many
    if np.size(a1) > 2 or np.size(a2) > 2:
        raise RuntimeError('intersection_linesegs takes 1 x 2 array as inputs for the first pair of linesegments.')

    if np.size(b1) == 2:
        # There is just one pair of linesegs to check, ie N = 1.
        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1
        dap = perp(da)
        denom = np.dot(dap, db)
        num = np.dot(dap, dp)
        intersect = (num / denom.astype(float)) * db + b1
        # print 'a1 = ', a1
        # print 'a2 = ', a2
        # print 'intersect = ', intersect
        ptok = point_is_on_linesegment_2D_singlept(intersect, a1, a2, thres=thres)
        # print 'ptok = ', ptok
        if ptok:
            ptok = point_is_on_linesegment_2D_singlept(intersect, b1, b2, thres=thres)
            # print 'ptok_try2 = ', ptok

        if ptok:
            return intersect, np.array([True])
        else:
            return None, np.array([False])
    else:
        # There are many pairs of linesegs to check against (a1, a2), ie N > 1.
        # Solve for the position s such that b1 + P(s) is orthogonal to dap:
        #       a1
        # ww    o-----> vvt
        #    /  |
        #  o    |
        # b1 \  |
        #      \|
        #       |\
        #       o  \
        #       a2  o b2
        da = a2 - a1
        db = b2 - b1
        ww = a1 - b1
        vvt = perp(da)
        num = np.dot(vvt, ww.T)
        denom = np.dot(vvt, db.T)
        # intersect = np.array([(num[ii] / denom[ii]) * db[ii] + b1[ii] for ii in range(len(b1))])
        intersect = (num / denom).reshape(len(num), 1) * np.ones_like(b1) * db + b1
        # print 'a1 = ', a1
        # print 'a2 = ', a2
        # print 'b1 = ', b1
        # print 'b2 = ', b2
        # print 'da = ', da
        # print 'db = ', db
        # print 'ww = ', ww
        # print 'num = ', num
        # print 'denom = ', denom
        # print 'vvt = ', vvt
        # print 'many pairs of linesegs to check:'
        # print 'intersect = ', intersect
        # plt.plot(intersect[:, 0], intersect[:, 1], 'kx')
        # print '\nchecking if intersect is on (a1, a2)'
        ptok_ab = point_is_on_linesegment_2D(intersect, a1, a2, thres=thres)
        # print '\nchecking if intersect is on (b1, b2)'
        ptok_ba = point_is_on_linesegment_2D(intersect, b1, b2, thres=thres)
        ptok = np.logical_and(ptok_ab, ptok_ba)
        # print 'ptok = ', ptok
        return intersect[np.where(ptok)[0]], ptok


def find_intersections(A, B):
    """Get the intersections between a sequence of connected line segments A and connected line segments B,
    without a nested for loop over linesegs. NOT FULLY VETTED!
    See linesegs_intersect_linesegs() instead.
    http://stackoverflow.com/questions/3252194/numpy-and-line-intersections

    Parameters
    ----------
    A : N x 2 float array
        first set of linesegments
    B : M x 2 float array
        second set of linesegments

    Returns
    ----------
    x, y : P x 2 float array
        coords of intersections
    """
    # min, max and all for arrays
    amin = lambda x1, x2: np.where(x1 < x2, x1, x2)
    amax = lambda x1, x2: np.where(x1 > x2, x1, x2)
    aall = lambda abools: np.dstack(abools).all(axis=2)
    slope = lambda line: (lambda d: d[:, 1] / d[:, 0])(np.diff(line, axis=0))

    x11, x21 = np.meshgrid(A[:-1, 0], B[:-1, 0])
    x12, x22 = np.meshgrid(A[1:, 0], B[1:, 0])
    y11, y21 = np.meshgrid(A[:-1, 1], B[:-1, 1])
    y12, y22 = np.meshgrid(A[1:, 1], B[1:, 1])

    m1, m2 = np.meshgrid(slope(A), slope(B))
    m1inv, m2inv = 1 / m1, 1 / m2

    yi = (m1 * (x21 - x11 - m2inv * y21) + y11) / (1 - m1 * m2inv)
    xi = (yi - y21) * m2inv + x21

    xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12),
              amin(x21, x22) < xi, xi <= amax(x21, x22))
    yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
              amin(y21, y22) < yi, yi <= amax(y21, y22))

    return xi[aall(xconds)], yi[aall(yconds)]


def linesegs_intersect_linesegs(lsa, lsb, thres=1e-7):
    """Determine if each lineseg in lsb intersects any linesegs in lsb, as efficiently as possible.
    Note that this is quasi-optimized for cases in which most linesegments in lsa are NOT near linesegments in lsb.
    The implementation is:
    1. check if the linesegments are even close to each other, then
    2. for each seg "ii" in lsa that is close to some in lsb, iterate over those in lsb that are close serially
       to check if it intersects the nearby seg in lsb, then
    3. for each seg in lsb nearby to seg "ii" that "ii" intersects, check if it intersects "ii"
    If 1, 2, and 3 are all true, then there is a true intersection.
    Example usage would have many linesegments lsa checked against a few linesegs lsb.
    reference for Java version: https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/

    Parameters
    ----------
    lsa : N x 4 float array
        set of linesegments to check for intersections. [[x0_a, y0_a, x0_b, y0_b], ..., [xi_a, yi_a, xi_b, yi_b], ...]]
    lsb : M x 4 float array
        second set of linesegments, which can be intersected by the tested set, lsa.
        Format: [[x0_a, y0_a, x0_b, y0_b], ..., [xi_a, yi_a, xi_b, yi_b], ...]]
    thres : float
        resolution such that if linsegs come within thres of each other, they are considered to intersect

    Returns
    ----------
    does_intersect : N x 1 bool array
        Whether the linesegments lsa intersect with any in lsb
    """
    # if lsb is a single lineseg, use different function
    if len(np.shape(lsb)) < 2:
        tmp, does_intersect = intersection_linesegs(lsb[0:2], lsb[2:4], lsa[:, 0:2], lsa[:, 2:4], thres=thres)
    else:
        # Check if bounding boxes intersect (returns and N x M bool array)
        bbi = bounding_boxes_intersect(lsa, lsb)
        # print 'bbi = ', bbi
        does_intersect = np.zeros(len(lsa), dtype=bool)
        # iterate serially over each line segment
        for ii in np.arange(len(lsa)):
            # iterate serially over each line segment whose bbox overlaps with that of ii
            bbok = np.where(bbi[ii])[0].ravel()
            # print '\nbbok = ', bbok
            # print 'endpt a1 = ', lsa[ii, 0:2]
            # print 'endpt a2 = ', lsa[ii, 2:4]
            # print 'endpts b1 = ', lsb[bbok, 0:2]
            # print 'endpts b2 = ', lsb[bbok, 2:4]
            if len(bbok) == 1:
                # For this single linseg of a (ii), see if it intersects any of the nearby linesegs in b
                ints, ints_bool = \
                    intersection_linesegs(lsa[ii, 0:2], lsa[ii, 2:4], lsb[bbok[0], 0:2], lsb[bbok[0], 2:4], thres=thres)
                does_intersect[ii] = ints_bool.any()
            elif len(bbok) > 0:
                # For this single linseg of a (ii), see if it intersects any of the nearby linesegs in b
                print 'line_segments: multiple comparison segments:'
                # print 'lsb[bbok, 0:2] = ', lsb[bbok, 0:2]
                ints, ints_bool = \
                    intersection_linesegs(lsa[ii, 0:2], lsa[ii, 2:4], lsb[bbok, 0:2], lsb[bbok, 2:4], thres=thres)
                does_intersect[ii] = ints_bool.any()

    return does_intersect


def point_is_on_linesegment_2D_singlept(p, a, b, thres=1e-5):
    """Check if point is on line segment (or vertical line is on plane in 3D, since 3rd dim ignored).
    See point_is_on_linesegment_2D for a more general handling of this functionality.

    Parameters
    ----------
    p : array or list of length >=2
        The point in 2D
    a : array or list of length >=2
        One end of the line segment
    b : array or list of length >=2
        The other end of the line segment
    thres : float
        How close must the point be to the line segment to be considered to be on it

    Returns
    ----------
    Boolean : whether the pt is on the line segment
    """
    crossproduct = (p[1] - a[1]) * (b[0] - a[0]) - (p[0] - a[0]) * (b[1] - a[1])
    if abs(crossproduct) > thres: return False  # (or != 0 if using integers)

    dotproduct = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
    if dotproduct < 0: return False

    squaredlengthba = (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])
    if dotproduct > squaredlengthba: return False

    return True


def point_is_on_linesegment_2D(p, a, b, thres=1e-5):
    """Check if point(s) lie on a line segment(s). Many points OR many linesegs are allowed, but not both.
    (or check if vertical line is on plane in 3D, since 3rd dim is ignored).

    Parameters
    ----------
    p : array of dimension #points x 2
        The points in 2D (or 3D with 3rd dim ignored)
    a : array or list of dimension #linesegs x 2
        One end of the line segment
    b : array or list of dimension #linesegs x 2
        The other end of the line segment
    thres : float
        How close must the point be to the line segment to be considered to be on it

    Returns
    ----------
    onseg: #points x 1 or #linesegs x 1 boolean array
        whether the pts are on the line segment
    """
    # Four cases:
    # (1) there is only one point to check, and only one linesegment
    # (2) one point, many linesegs
    # (3) many points, one lineseg
    # (4) N points, N linsegs, to check against each other one-to-one
    # (else) return an error
    if np.size(p) == 2 and np.size(a) == 2 and np.size(b) == 2:
        print 'single pts - single lineseg comparison'
        crossproduct = (p[1] - a[1]) * (b[0] - a[0]) - (p[0] - a[0]) * (b[1] - a[1])
        if abs(crossproduct) > thres: return np.array([False])  # (or != 0 if using integers)

        dotproduct = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
        if dotproduct < 0: return np.array([False])

        squaredlengthba = (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])
        if dotproduct > squaredlengthba: return np.array([False])

        return np.array([True])
    elif np.size(a) == 2 and np.size(b) == 2:
        print 'pt is Nx1, one lineseg to test'
        # print 'p = ', p
        # print 'a = ', a
        # print 'b = ', b
        crossproduct = (p[:, 1] - a[1]) * (b[0] - a[0]) - (p[:, 0] - a[0]) * (b[1] - a[1])
        dotproduct = (p[:, 0] - a[0]) * (b[0] - a[0]) + (p[:, 1] - a[1]) * (b[1] - a[1])
        squaredlengthba = (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])
        on_seg = np.logical_and(np.logical_and(abs(crossproduct) < thres, dotproduct > 0),
                                dotproduct*np.ones(len(p[:,0]), dtype=float) < squaredlengthba)
        # print 'crossproduct = ', crossproduct
        # print 'dotproduct = ', dotproduct
        # print 'squaredlengthba = ', squaredlengthba
        # print 'abs(crossproduct) < thres = ', abs(crossproduct) < thres
        # print 'dotproduct > 0 = ', dotproduct > 0
        # print 'dotproduct*np.ones(len(p[:,0]), dtype=float) < squaredlengthba = ', \
        #     dotproduct*np.ones(len(p[:,0]), dtype=float) < squaredlengthba
        # print 'on_seg = ', on_seg
        return on_seg
    elif np.size(p) == 2:
        print 'pt is 2x1, many linesegs to test'
        crossproduct = (p[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) - (p[0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        dotproduct = (p[0] - a[:, 0]) * (b[:, 0] - a[:, 0]) + (p[1] - a[:, 1]) * (b[:, 1] - a[:, 1])
        squaredlengthba = (b[:, 0] - a[:, 0]) * (b[:, 0] - a[:, 0]) + (b[:, 1] - a[:, 1]) * (b[:, 1] - a[:, 1])
        on_seg = np.logical_and(np.logical_and(abs(crossproduct) < thres, dotproduct > 0),
                                dotproduct < squaredlengthba)
        return on_seg
    elif np.size(p) == np.size(a):
        print 'pt is Nx2, linesegs are Nx2, so check iith point against iith lineseg'
        # pt is Nx2, linesegs are Nx2, so check iith point against iith lineseg
        crossproduct = (p[:, 1] - a[:, 1]) * (b[:, 0] - a[:, 0]) - (p[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        dotproduct = (p[:, 0] - a[:, 0]) * (b[:, 0] - a[:, 0]) + (p[:, 1] - a[:, 1]) * (b[:, 1] - a[:, 1])
        squaredlengthba = (b[:, 0] - a[:, 0]) * (b[:, 0] - a[:, 0]) + (b[:, 1] - a[:, 1]) * (b[:, 1] - a[:, 1])
        on_seg = np.logical_and(np.logical_and(abs(crossproduct) < thres, dotproduct > 0),
                                dotproduct < squaredlengthba)
        return on_seg


def closest_pt_along_line(pt, endpt1, endpt2):
    """Get point along a line defined by two points (endpts), closest to a point not on the line

    Parameters
    ----------
    pt : array of length 2
        point near which to find nearest point
    endpt1, endpt2 : arrays of length 2
        x,y positions of points on line as array([[x0,y0],[x1,y1]])

    Returns
    ----------
    proj : array of length 2
        the point nearest to pt along line
    """
    #     .pt   /endpt2
    #          /
    #        7/proj
    #       //
    #      //endpt1
    #
    # v is vec along line seg
    a = endpt2[0] - endpt1[0]
    b = endpt2[1] - endpt1[1]
    x = pt[0] - endpt1[0]
    y = pt[1] - endpt1[1]

    # the projection of the vector to pt along v (no numpy)
    p = np.array(
        [a * (a * x + b * y) / (a ** 2 + b ** 2) + endpt1[0], b * (a * x + b * y) / (a ** 2 + b ** 2) + endpt1[1]])
    # print 'p (in closest_pt_along...) =', p
    return p


def closest_pts_along_line(pts, endpt1, endpt2):
    """For each coordinate in pts, get point along a line defined by two points (endpts) which is closest to that
    coordinate. Returns p as numpy array of points along the line. Right now just works in 2D.

    Parameters
    ----------
    pts : NP x 2 float array
        point near which to find nearest point
    endpt1, endpt2 : 2 x 1 float arrays
        x,y positions of points on line as array([[x0,y0],[x1,y1]])

    Returns
    ----------
    proj : array of length 2
        the points along line that are nearest to each coordinate in pts
    """
    # if 2D
    # v is vec along line seg
    aa = endpt2[0] - endpt1[0]
    bb = endpt2[1] - endpt1[1]
    xx = pts[:, 0] - endpt1[0]
    yy = pts[:, 1] - endpt1[1]
    # the projection of the vector to pt along v
    projv = np.dstack((aa * (aa * xx + bb * yy) / (aa ** 2 + bb ** 2), bb * (aa * xx + bb * yy) / (aa ** 2 + bb ** 2)))[
        0]
    # add the endpt whose position was subtracted
    pp = projv + endpt1 * np.ones(projv.shape)
    # else if 3D:
    # todo : generalize to 3d
    return pp


def closest_pt_on_lineseg(pt, endpts):
    """Get point on line segment closest to a point not on the line; could be an endpt if lineseg is distant from pt.

    Parameters
    ----------
    pt : array of length 2
        point near which to find near point
    endpts : array of dimension 2x2
        x,y positions of endpts of line segment as array([[x0,y0],[x1,y1]])

    Returns
    ----------
    pt : array of length 2
        the point nearest to pt on lineseg
    """
    p = closest_pt_along_line(pt, endpts)
    d0 = np.sqrt((proj[1] - pt[1]) ** 2 + (proj[0] - pt[0]) ** 2)
    d1 = np.sqrt((endpts[0, 1] - pt[1]) ** 2 + (endpts[0, 0] - pt[0]) ** 2)
    d2 = np.sqrt((endpts[1, 1] - pt[1]) ** 2 + (endpts[1, 0] - pt[0]) ** 2)
    if d0 <= d1:
        if d0 <= d2:
            return p
        else:
            return endpts[1, :]
    else:
        if d1 < d2:
            return endpts[0, :]
        else:
            return endpts[1, :]


def closest_pts_on_lineseg(pts, endpt1, endpt2):
    """Get points on line segment closest to an array of points not on the line;
    the closest point can be an endpt, for example if the lineseg is distant from pt.
    Works in any dimension now, if closest_pts_along_line works in any dimension.

    Parameters
    ----------
    pts : N x dim float array
        points near which to find near point
    endpt1 : dim x 1 float array
        position of first endpt of line segment as array([x0, x1, ... xdim])
    endpt2 : dim x 1 float array
        position of second endpt of line segment as array([x0, x1, ... xdim])

    Returns
    ----------
    p : N x dim float array
        the point nearest to pt on lineseg
    d : float
        distance from pt to p
    """
    # create output vectors
    pout = np.zeros_like(pts)
    dout = np.zeros_like(pts[:, 0])

    # Find nearest p along line formed by endpts
    p = closest_pts_along_line(pts, endpt1, endpt2)
    d0 = np.linalg.norm(p - pts, axis=1)

    # is p ok?-- are they the line segment? or is out of bounds?
    pok = line_pts_are_on_lineseg(p, endpt1, endpt2)

    # Assign those pts and distances for pok indices
    pout[pok, :] = p[pok, :]
    dout[pok] = d0[pok]

    # For p not on the segment, pick closer endpt
    d1 = np.linalg.norm(endpt1 - pts, axis=1)
    d2 = np.linalg.norm(endpt2 - pts, axis=1)

    nd1 = d1 < d2  # nearer to d1
    ntd1 = np.logical_and(~pok, nd1)  # nearest to d1
    ntd2 = np.logical_and(~pok, ~nd1)  # nearest to d2

    pout[ntd1, :] = endpt1
    dout[ntd1] = d1[ntd1]
    pout[ntd1, :] = endpt2
    dout[ntd2] = d2[ntd2]

    return pout, dout


def closest_pts_on_lineseg_2D(pts, endpt1, endpt2):
    """Get points on line segment closest to an array of points not on the line;
    the closest point can be an endpt, for example if the lineseg is distant from pt.

    Parameters
    ----------
    pts : array N x 2
        points near which to find near point
    endpt1 : 2x2 float array
        x,y positions of endpts of line segment as array([[x0,y0],[x1,y1]])
    endpt2 : 2x2 float array
        x,y positions of endpts of line segment as array([[x0,y0],[x1,y1]])

    Returns
    ----------
    p : array of length 2
        the point nearest to pt on lineseg
    d : float
        distance from pt to p
    """
    # create output vectors
    pout = np.zeros_like(pts)
    dout = np.zeros_like(pts[:, 0])

    # Find nearest p along line formed by endpts
    p = closest_pts_along_line(pts, endpt1, endpt2)
    d0 = np.sqrt((p[:, 1] - pts[:, 1]) ** 2 + (p[:, 0] - pts[:, 0]) ** 2)

    # is p ok?-- are they the line segment? or is out of bounds?
    pok = line_pts_are_on_lineseg(p, endpt1, endpt2)

    # Assign those pts and distances for pok indices
    pout[pok, :] = p[pok, :]
    dout[pok] = d0[pok]

    # For p not on the segment, pick closer endpt
    d1 = (endpt1[1] - pts[:, 1]) ** 2 + (endpt1[0] - pts[:, 0]) ** 2
    d2 = (endpt2[1] - pts[:, 1]) ** 2 + (endpt2[0] - pts[:, 0]) ** 2

    nd1 = d1 < d2  # nearer to d1
    ntd1 = np.logical_and(~pok, nd1)  # nearest to d1
    ntd2 = np.logical_and(~pok, ~nd1)  # nearest to d2

    pout[ntd1, :] = endpt1
    dout[ntd1] = np.sqrt(d1[ntd1])
    pout[ntd1, :] = endpt2
    dout[ntd2] = np.sqrt(d2[ntd2])

    return pout, dout


def line_pts_are_on_lineseg(p, a, b):
    """Check if an array of n-dimensional points (p) which lie along a line is between two other points (a,b) on
    that line (ie, is on a line segment)

    Parameters
    ----------
    p : array of dim N x 2
        points for which to evaluate if they are on segment
    a,b : dim x 1 float arrays or lists
        positions of line segment endpts

    Returns
    ----------
    True or False: whether pt is between endpts
    """
    # dot product must be positive and less than |b-a|^2
    if np.shape(p)[1] == 2:
        dotproduct = (p[:, 0] - a[0]) * (b[0] - a[0]) + (p[:, 1] - a[1]) * (b[1] - a[1])
        squaredlengthba = (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])
    else:
        # Do n-dimensional dot product:
        dotproduct = np.zeros_like(p[:, 0])
        for dim in range(np.shape(p)[1]):
            dotproduct += (p[:, dim] - a[dim]) * (b[dim] - a[dim])
        squaredlengthba = np.linalg.norm(b - a, axis=1) ** 2
    return np.logical_and(dotproduct > 0, dotproduct < squaredlengthba)


def line_pts_are_on_lineseg_2D(p, a, b):
    """Check if an array of points (p) which lie along a line is between two other points (a,b) on that line (ie, is
    on a line segment)

    Parameters
    ----------
    p : array of dim N x 2
        points for which to evaluate if they are on segment
    a,b : arrays or lists of length 2
        x,y positions of line segment endpts

    Returns
    ----------
    True or False: whether pt is between endpts
    """
    # dot product must be positive and less than |b-a|^2
    dotproduct = (p[:, 0] - a[0]) * (b[0] - a[0]) + (p[:, 1] - a[1]) * (b[1] - a[1])
    squaredlengthba = (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])
    return np.logical_and(dotproduct > 0, dotproduct < squaredlengthba)


def pts_are_near_lineseg(x, endpt1, endpt2, W):
    """Determine if pts in array x are within W of line segment

    Parameters
    ----------
    x : NP x 2 float array
        the points to consider proximity to a lineseg
    endpt1 : 2x2 float array
        the first endpoint of the linesegment
    endpt2 : 2x2 float array
        the second endpoint of the linesegment
    W : float
        threshold distance from lineseg to consider a point 'close'

    Returns
    -------
    NP x 1 boolean array
        Whether the points are closer than W from the linesegment
    """
    # check if point is anywhere near line before doing calcs
    p, dist = closest_pts_on_lineseg(x, endpt1, endpt2)
    return dist <= W


def mindist_from_multiple_linesegs(pts, linesegs):
    """Return the minimum distance between an array of points and any point lying on any of the linesegments
    in the given list of linesegments 'linesegs'.

    Parameters
    ----------
    pts : Nx2 array (or list?)
        x,y positions of points
    linesegs : Nx4 array or list
        each row contains x,y of start point, x,y of end point

    Returns
    ---------
    dist : float
        minimum distance to any point lying on any of the linesegs
    """
    first = True
    # print 'pts  = ', pts
    # print 'linesegs = ', linesegs
    for row in linesegs:
        endpt1 = [row[0], row[1]]
        endpt2 = [row[2], row[3]]
        p, dist0 = closest_pts_on_lineseg(pts, endpt1, endpt2)
        if first:
            dist = dist0
            first = False
        else:
            # print 'shape(dist) = ', np.shape(dist)
            # print 'shape(dist0) = ', np.shape(dist0)
            dist = np.min(np.vstack((dist, dist0)), axis=0)
            # print 'dist = ', dist
            # print 'shape(dist) = ', np.shape(dist)
    return dist


def array_to_linesegs(xypts):
    """Convert an array of xy points into an array of consecutive (connected) linesegments (where each row contains
    x0, y0, x1, y1).

    Parameters
    ----------
    xypts : N x 2 float array
        2d coordinates to connect together into consecutive linesegments

    Returns
    -------
    linesegs : Nx4 array or list
        each row contains x,y of start point, x,y of end point
    """
    # If the starting point is included twice (at beginning and end), then chop it from the end
    if (xypts[0] == xypts[-1]).all():
        xypts = xypts[0:-1]

    # Create the linesegment array
    linesegs = np.zeros((len(xypts), 4), dtype=np.float)
    linesegs[:, 0:2] = xypts
    linesegs[:, 2:4] = np.roll(xypts, -1, axis=0)
    return linesegs


def xyBL2linesegs(xy, BL):
    """Form linesegments array from positions and bond definitions

    Parameters
    ----------
    xy : n x 2 float array
        the positions of all particles to convert to linesegment endpoints.
        The connectivity of the linesegs from these points are determined by BL
    BL : m x 2 int array
        BL[i] gives the ith 'bond', which is converted to a linesegment (a 1 x 4 float array in format x0 y0 x1 y1)
        The connectivity of the points xy to convert to linesegments

    Returns
    -------
    linesegs :  len(BL) x 4 array or list
        each row contains x,y of start point, x,y of end point
    """
    linesegs = np.zeros((len(BL), 4), dtype=float)
    ii = 0
    for bond in BL:
        linesegs[ii, :] = np.hstack((xy[bond[0]], xy[bond[1]]))
        ii += 1

    return linesegs


if __name__ == '__main__':
    """Here we demonstrate some functionality of this module"""
    import argparse

    parser = argparse.ArgumentParser('Demonstrate some functions in line_segments.py')
    parser.add_argument('-demo_intersections', '--demo_intersections', help='do demo with intersecting linesegs',
                        action='store_true')
    args = parser.parse_args()

    if args.demo_intersections:
        import matplotlib.mlab as mlab
        # Demonstrate linesegs_intersect_linesegs
        xya = np.random.rand(4, 4)*10
        xyb = np.random.rand(4, 4)*10
        print 'xya = ', xya
        print 'xyb = ', xyb
        does_intersect = linesegs_intersect_linesegs(xya, xyb)
        for ii in range(len(xya)):
            if does_intersect[ii]:
                plt.plot([xya[ii, 0], xya[ii, 2]], [xya[ii, 1], xya[ii, 3]], 'g.-')
            else:
                plt.plot([xya[ii, 0], xya[ii, 2]], [xya[ii, 1], xya[ii, 3]], 'b.-')
        for ii in range(len(xyb)):
            plt.plot([xyb[ii, 0], xyb[ii, 2]], [xyb[ii, 1], xyb[ii, 3]], 'r.-')
            plt.text(0.5*(xyb[ii, 0] + xyb[ii, 2]), 0.5*(xyb[ii, 1]+xyb[ii, 3])+0.5, str(ii) )
        plt.show()

        # Demonstrate intersection_linesegs
        a1 = np.array([-1, 0])
        a2 = np.array([3, 0])
        b1 = np.array([[2, -1], [1.5, -1]])
        b2 = np.array([[4, 1], [1.5, 1]])
        ints, doint = intersection_linesegs(a1, a2, b1, b2, thres=1e-6)
        print 'ints = ', ints
        print 'np.array(ints) = ', np.array(ints)
        plt.plot([a1[0], a2[0]], [a1[1], a2[1]], 'b.-')
        for ii in range(len(b1)):
            plt.plot([b1[ii, 0], b2[ii, 0]], [b1[ii, 1], b2[ii, 1]], 'r.-')
        for ii in range(len(ints[:, 0])):
            plt.scatter(ints[ii][0], ints[ii][1], c='g')
        plt.show()

        # Demonstrate find_intersections
        A = np.array([[-1, 1], [1, 1], [1, 0], [2, 0]])
        B = np.array([[0, 0], [0, 2], [1.5, -1], [2, 1]])
        ints = find_intersections(A, B)
        plt.plot(A[:, 0], A[:, 1], 'b-')
        plt.plot(B[:, 0], B[:, 1], 'r-')
        plt.scatter(ints[0][0], ints[1][0], c='b')
        plt.scatter(ints[0][1], ints[1][1], c='r')
        print 'ints = ', ints
        print 'np.array(ints) =', np.array(ints)
        plt.show()

        # Use a contour plot
        delta = 0.025
        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-2.0, 2.0, delta)
        X, Y = np.meshgrid(x, y)
        Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
        Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
        # difference of Gaussians
        Z = 10.0 * (Z2 - Z1)

        # Create a simple contour plot with labels using default colors.  The
        # inline argument to clabel will control whether the labels are draw
        # over the line segments of the contour, removing the lines beneath
        # the label
        plt.figure()
        Acs = plt.contour(X, Y, Z)
        Bcs = plt.contour(Y, X + 1, Z)
        plt.clf()

        # A and B are the two lines, each is a
        # two column matrix
        A = Acs.collections[0].get_paths()[0].vertices
        B = Bcs.collections[0].get_paths()[0].vertices
        print 'A = ', A
        print 'B = ', B

        ints = find_intersections(A, B)
        print 'ints = ', ints
        print 'np.array(ints) =', np.array(ints)
        plt.plot(A[:, 0], A[:, 1], 'b-')
        plt.plot(B[:, 0], B[:, 1], 'r-')
        plt.scatter(ints[0][0], ints[1][0], c='b')
        plt.scatter(ints[0][1], ints[1][1], c='r')
        plt.show()
