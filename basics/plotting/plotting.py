import basics.plotting.colormaps as lecmaps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm
import basics.plotting.science_plot_style as sps

"""General functions for plotting"""


def golden_ratio():
    """Return the golden ratio for aethetically pleasing axis sizing"""
    return 1.61803398875


def set_fontsizes(sizes=None):
    """Set all matplotlib font sizes

    Parameters
    ----------
    sizes : int or tuple of 1 or 3 ints or None
        fontsizes for (default, axes title, xtick label, ytick label, legend) for small, axes label for medium,
        title for large
    """
    if sizes is None:
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12
    elif isinstance(sizes, int):
        SMALL_SIZE = sizes
        MEDIUM_SIZE = sizes
        BIGGER_SIZE = sizes
    else:
        SMALL_SIZE = sizes[0]
        MEDIUM_SIZE = sizes[1]
        BIGGER_SIZE = sizes[2]

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


def plot_pcolormesh_scalar(x, y, C, outpath, title, xlabel=None, ylabel=None, title2='', subtext='', subsubtext='',
                           vmin='auto', vmax='auto', cmap="coolwarm", show=False, close=True, axis_on=True, FSFS=20,
                           ax=None, zorder=None):
    """Save a single-panel plot of a scalar quantity C as colored pcolormesh

    Parameters
    ----------
    x, y : NxN mesh arrays
        the x and y positions of the points evaluated to Cx, Cy
    C : NxN arrays
        values for the plotted quantity C evaluated at points (x,y)
    outpath : string
        full name with file path
    title : string
        title of the plot
    title2 : string
        placed below title
    subtext : string
        placed below plot
    subsubtext : string
        placed at bottom of image
    vmin, vmax : float
        minimum, maximum value of C for colorbar; default is range of values in C
    cmap : matplotlib colormap
    show : bool
        whether to display the plot for interactive viewing
    close : bool
        whether to close the plot at end of function
    axis_on : bool
        if False, axis labels will be removed
    """
    if (cmap == 'coolwarm' or cmap == 'seismic') and (vmin == 'auto' and vmax == 'auto'):
        # symmetric colormaps call for symmetric limits
        vmax = np.max(np.abs(C.ravel()))
        vmin = - vmax
    if isinstance(vmin, str):
        vmin = np.nanmin(C)
    if isinstance(vmax, str):
        vmax = np.nanmax(C)

    if ax is None:
        plt.close('all')
        fig, ax = plt.subplots(1, 1)
    else:
        fig = plt.gcf()

    # scatter scale (for color scale)
    scsc = ax.pcolormesh(x, y, C, cmap=cmap, vmin=vmin, vmax=vmax, zorder=zorder)
    ax.set_aspect('equal')
    if not axis_on:
        ax.axis('off')
    ax.set_title(title, fontsize=FSFS)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FSFS)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FSFS)
    fig.text(0.5, 0.12, subtext, horizontalalignment='center')
    fig.text(0.5, 0.05, subsubtext, horizontalalignment='center')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scsc, cax=cbar_ax)
    fig.text(0.5, 0.98, title2, horizontalalignment='center', verticalalignment='top')

    if outpath is not None and outpath != 'none':
        print('outputting matrix image to ', outpath)
        plt.savefig(outpath + '.png')
    if show:
        plt.show()
    if close:
        plt.close()
    else:
        return fig, ax


def plot_real_matrix(M, name='', outpath=None, fig='auto', climv=None, cmap="coolwarm", show=False, close=True,
                     fontsize=None):
    """Plot matrix as colored subplot, with red positive and blue negative.

    Parameters
    ----------
    M : complex array
        matrix to plot
    name : string
        name to save plot WITHOUT extension (png)
    outpath : string (default='none' -> no saving)
        Directory and name of file as which to save plot. If outpath is None or 'none', does not save plot.
    show : bool (default == False)
        Whether to show the plot (and force user to close it to continue)
    clear : bool (default == True)
        Whether to clear the plot after saving or showing
    Returns
    ----------

    """
    if climv is None:
        climv = np.max(np.abs(M.ravel()))
        climvs = (-climv, climv)
    else:
        climvs = climv

    if fig == 'auto':
        fig = plt.gcf()
        plt.clf()

    a = fig.add_subplot(1, 1, 1)
    img = a.imshow(M, cmap="coolwarm", interpolation='none')
    a.set_title(name, fontsize=fontsize)
    cbar = plt.colorbar(img, orientation='horizontal')
    cbar.set_clim(climvs)

    if outpath is not None and outpath != 'none':
        print('outputting matrix image to ', outpath)
        plt.savefig(outpath + '.png')
    if show:
        plt.show()
    if close:
        plt.clf()


def plot_complex_matrix(M, name='', outpath=None, fig='auto', climvs=[], show=False, close=True, fontsize=None):
    """Plot real and imaginary parts of matrix as two subplots

    Parameters
    ----------
    M : complex array
        matrix to plot
    name : string
        name to save plot WITHOUT extension (png)
    outpath : string (default='none' -> no saving)
        Directory and name of file as which to save plot. If outpath is None or 'none', does not save plot.
    fig : matplotlib figure instance
        The figure to use for the plots
    clims : list of two lists
        Real and imaginary plot colorlimits, as [[real_lower, real_upper], [imag_lower, imag_upper]]
    show : bool (default == False)
        Whether to show the plot (and force user to close it to continue)
    close : bool (default == True)
        Whether to clear the plot after saving or showing
    fontsize : int
        The font size for the title, if name is not empty

    Returns
    ----------

    """
    # unpack or set colorlimit values
    if not climvs:
        climv_real_lower = -np.max(np.abs(np.real(M).ravel()))
        climv_real_upper = np.max(np.abs(np.real(M).ravel()))
        climv_imag_lower = -np.max(np.abs(np.imag(M).ravel()))
        climv_imag_upper = np.max(np.abs(np.imag(M).ravel()))
    else:
        climv_real_lower = climvs[0][0]
        climv_real_upper = climvs[0][1]
        climv_imag_lower = climvs[1][0]
        climv_imag_upper = climvs[1][1]

    if fig == 'auto':
        fig = plt.gcf()
        plt.clf()

    ax1 = fig.add_subplot(1, 2, 1)
    img = ax1.imshow(np.imag(M), cmap="coolwarm", interpolation='none')
    ax1.set_title('Imaginary part ' + name, fontsize=fontsize)
    cbar1 = plt.colorbar(img, orientation='horizontal')
    cbar1.set_clim(climv_imag_lower, climv_imag_upper)

    ax2 = fig.add_subplot(1, 2, 2)
    img2 = ax2.imshow(np.real(M), cmap="coolwarm", interpolation='none')
    ax2.set_title('Real part ' + name, fontsize=fontsize)
    cbar2 = plt.colorbar(img2, orientation='horizontal')
    cbar2.set_clim(climv_real_lower, climv_real_upper)

    if outpath is not None and outpath != 'none':
        print('outputting complex matrix image to ', outpath)
        plt.savefig(outpath + '.png')
    if show:
        plt.show()
    if close:
        plt.clf()
    return fig, (ax1, ax2), (cbar1, cbar2)


def absolute_sizer(ax=None):
    """Use the size of the matplotlib axis to create a function for sizing objects"""
    ppu = get_points_per_unit(ax)
    return lambda x: np.pi * (x * ppu) ** 2


def empty_scalar_mappable(vmin, vmax, cmap):
    """Create a scalar mappable for creating a colorbar that is not linked to specific data.

    Examples
    --------
    sm = empty_scalar_mappable(-1.0, 1.0, 'viridis')
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation=cbar_orientation, format=cbar_tickfmt)
    cbar.set_label(cax_label, labelpad=cbar_labelpad, rotation=0, fontsize=fontsize, va='center')

    Parameters
    ----------
    vmin : float
        the lower limit of the mappable
    vmax : float
        the upper limit of the mappable
    cmap : matplotlib colormap
        the colormap to use for the mappable
    """
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    return sm


def get_markerstyles(n=None):
    """Get a list of n marker style keys for matplotlib marker='' arguments, in a nicely defined order (so bizarre
    markers appear only for large n).
    If n < 28 (which is the number of 'nice' build-in markers, meaning ones I deem reasonably suitable for a plot,
    then all are distinct. Otherwise, some repeat."""
    all_markers = ['o', 'D', 's', '2', '*', 'h', '8', 'v', 'x', '+', 5, 'd', '>', 7, '.', '1', 'p', '3',
                   6, 0, 1, 2, 3, 4, '4', '<', 'H', '^']
    # Note: 0: 'tickleft', 1: 'tickright', 2: 'tickup', 3: 'tickdown', 4: 'caretleft', 'D': 'diamond', 6: 'caretup',
    #  7: 'caretdown', 's': 'square', '|': 'vline', '': 'nothing', 'None': 'nothing', 'x': 'x', 5: 'caretright',
    #  '_': 'hline', '^': 'triangle_up', ' ': 'nothing', 'd': 'thin_diamond', 'h': 'hexagon1', '+': 'plus', '*': 'star',
    #  ',': 'pixel', 'o': 'circle', '.': 'point', '1': 'tri_down', 'p': 'pentagon', '3': 'tri_left', '2': 'tri_up',
    #  '4': 'tri_right', 'H': 'hexagon2', 'v': 'triangle_down', '8': 'octagon', '<': 'triangle_left', None: 'nothing',
    #  '>': 'triangle_right'
    # all_markers = ['circle', 'diamond', 'square', 'tri_up', 'star',
    #                'hexagon1', 'octagon', 'triangle_down', 'x', 'plus',
    #                'caretright', 'thin_diamond', 'triangle_right', 'caretdown',
    #                'point', 'tri_down', 'pentagon', 'tri_left', 'caretup',
    #                'tickleft', 'tickright', 'tickup', 'tickdown', 'caretleft',
    #                'tri_right', 'triangle_left', 'hexagon2', 'triangle_up']
    # Note: markers can be found via
    # import matplotlib.pyplot as plt
    # import matplotlib
    # d = matplotlib.markers.MarkerStyle.markers
    # def find_symbol(syms):
    #     outsyms = []
    #     for sym in syms:
    #         for key in d:
    #             if d[key] == sym:
    #                 outsyms.append(key)
    #     return outsyms

    if n is None:
        markerlist = all_markers
    elif n > len(all_markers):
        markerlist = all_markers
        markerlist.append(all_markers[0:n - len(all_markers)])
    else:
        markerlist = all_markers[0:n]

    return markerlist


def add_at(ax, t, loc=2):
    """Add attribute to a makeshift legend
    """
    from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
    fp = dict(size=8)
    _at = AnchoredText(t, loc=loc, prop=fp)
    ax.add_artist(_at)
    return _at


def annotate_connection_style(ax, x1y1, x2y2, connectionstyle="angle3,angleA=0,angleB=90",
                              xycoords='figure_fraction', textcoords='figure_fraction', color="0.5",
                              label=None):
    """Draw an annotation with a prescribed connection style on an axis.
    Example usage:
    demo_con_style(column[1], "angle3,angleA=0,angleB=90",
               label="angle3,\nangleA=0,\nangleB=90")
    See http://matplotlib.org/users/annotations_guide.html

    Parameters
    ----------
    ax : axis instance
        The axis on which to annotate
    x1y1 : tuple of floats
        coordinate pointed from
    x2y2 : tuple of floats
        coordinate pointed to
    connectionstyle : str
        Specifier for connection style, ex: "angle3,angleA=0,angleB=90"
    xycoords : str ('figure_fraction', 'axis_fraction', 'data')
        How to measure the location of the xy coordinates
    textcoords : str ('figure_fraction', 'axis_fraction', 'data')
        How to measure the location of the text coordinates
    color : color spec
        color of the arrow
    label : str or None
        label of annotation for a legend
    """
    if label is not None:
        add_at(ax, label, loc=2)

    x1, y1 = x1y1
    x2, y2 = x2y2

    ax.annotate("",
                xy=(x1, y1), xycoords=xycoords,
                xytext=(x2, y2), textcoords=textcoords,
                arrowprops=dict(arrowstyle="simple",  # linestyle="dashed",
                                color=color,
                                shrinkA=5,
                                shrinkB=5,
                                patchA=None,
                                patchB=None,
                                connectionstyle=connectionstyle,
                                ),
                )


def plot_pcolormesh(x, y, z, n, ax=None, cax=None, method='nearest', make_cbar=True, cmap='viridis',
                    vmin=None, vmax=None, title=None, xlabel=None, ylabel=None, ylabel_right=True,
                    ylabel_rot=90, cax_label=None, cbar_labelpad=3, cbar_orientation='vertical',
                    ticks=None, cbar_nticks=None, fontsize=12, title_axX=None, title_axY=None, alpha=1.0,
                    zorder=None):
    """Interpolate x,y,z data onto an nxn meshgrid and plot as heatmap

    Parameters
    ----------
    x : n x 1 float array
        the x coordinates of the unstructured data
    y : n x 1 float array
        the y coordinates of the unstructured data
    z : float or int array
        the float mapped to a color for each pixel
    n : int
        number of elements in each linear dimension in the meshgrid formed from x,y
    ax : axis instance or None
    cax : axis instance or None
        axis on which to plot the colorbar
    method : str
        interpolation specifier string
    make_cbar : bool
        Make a colorbar for the plot. If False, cax, cax_label, cbar_labelpad, cbar_orientation, and cbar_nticks are
        ignored
    cmap : colormap specifier
        The colormap to use for the pcolormesh
    vmin : float or None
        Color limit minimum value
    vmax : float or None
        Color limit maximum value
    title : str or None
        title for the plo
    titlepad : int or None
        Space above the plot to place the title
    alpha : float
        opacity
    """
    from lepm.data_handling import interpol_meshgrid
    X, Y, Z = interpol_meshgrid(x, y, z, n, method=method)
    if ax is None:
        ax = plt.gca()
    if cmap not in plt.colormaps():
        lecmaps.register_colormaps()
    pcm = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, zorder=zorder)

    if make_cbar:
        print('making colorbar in plot_pcolormesh()...')
        cbar = plt.colorbar(pcm, cax=cax, orientation=cbar_orientation)
        if ticks is not None:
            cbar.set_ticks(ticks)
        elif cbar_nticks is not None:
            cbar.set_ticks(np.linspace(np.min(Z.ravel()), np.max(Z.ravel()), cbar_nticks))
        if cax_label is not None:
            if cbar_orientation == 'vertical':
                cbar.set_label(cax_label, labelpad=cbar_labelpad, rotation=0, fontsize=fontsize, va='center')
            else:
                cbar.set_label(cax_label, labelpad=cbar_labelpad, rotation=0, fontsize=fontsize)

    if title is not None:
        print('\n\n\nplotting.plotting: Making title\n\n\n')
        if title_axX is None and title_axY is None:
            ax.set_title(title, fontsize=fontsize)
        elif title_axX is None:
            print('plotting.plotting: placing title at custom Y position...')
            ax.text(0.5, title_axY, title,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)
        elif title_axY is None:
            print('plotting.plotting: placing title at custom X position...')
            ax.text(title_axX, 1.0, title,
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    transform=ax.transAxes)
        else:
            print('plotting.plotting: placing title at custom XY position...')
            ax.text(title_axX, title_axY, title,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, rotation=ylabel_rot)
        if ylabel_right:
            ax.yaxis.set_label_position("right")

    return ax


def change_axes_geometry_stack(fig, ax, naxes):
    """Take a figure with stacked axes and add more stacked (in Y) axes to the figure, shuffling the others
    to make room for the new one(s).
    """
    for ii in range(len(ax)):
        geometry = (naxes, 1, ii + 1)
        if ax[ii].get_geometry() != geometry:
            ax[ii].change_geometry(*geometry)

    for ii in np.arange(len(ax), naxes):
        print('adding axis ', ii)
        fig.add_subplot(naxes, 1, ii + 1)

    ax = fig.axes
    return fig, ax


def plot_axes_on_fig(ax, fig=None, geometry=(1, 1, 1)):
    """Attach ax to a new or different figure"""
    if fig is None:
        fig = plt.figure()
    if ax.get_geometry() != geometry:
        ax.change_geometry(*geometry)
    axes = fig.axes.append(ax)
    return fig, axes



def initialize_portrait_with_header(preset_cbar=False, ax_pos=[0.1, 0.10, 0.8, 0.60],
                                    cbar_pos=[0.79, 0.80, 0.012, 0.15],
                                    ax_cbar_pos=[0.9, 0.4, 0.012, 0.15], ax_cbar=False):
    """

    Parameters
    ----------
    preset_cbar : bool
        If True, specifies colorbar position on plot using cbar_pos keywork argument to place the colorbar.
    orientation : str
        portrait or landscape, for figure aspect ratio
    cbar_pos : list of 4 floats
        xbottom ybottom, width and height of colorbar in figure_fraction coordinates
    kwargs : keyword arguments for colored_DOS_plot()
        For example, colorV = 1./ipr, DOSexcite=(2.25, 200.)

    Returns
    ----------
    fig : matplotlib figure
        figure with lattice and DOS drawn
    DOS_ax : axis instance
        axis for the DOS plot
    ax : axis instance
        axis for the time domain or other plots
    """
    fig = plt.figure(figsize=(1.5 * 5, 1.5 * 7))
    if preset_cbar:
        header_cbar = plt.axes(cbar_pos)
        header_ax = plt.axes([0.20, cbar_pos[1], 0.55, 0.15])
    else:
        header_ax = plt.axes([0.20, cbar_pos[1], 0.70, 0.15])
        header_cbar = None
    # axes constructor axes([left, bottom, width, height])
    ax = plt.axes(ax_pos)
    if ax_cbar:
        ax_cbar = plt.axes(ax_cbar_pos)
    else:
        ax_cbar = None

    return fig, ax, header_ax, header_cbar, ax_cbar


def initialize_portrait(ax_pos=[0.1, 0.10, 0.8, 0.60]):
    """Create a portrait-style figure

    Parameters
    ----------
    ax_pos : list of 4 floats
        The xstart, ystart, width, and height of the axis to draw on the portrait canvas

    Returns
    ----------
    fig : matplotlib figure
        figure with lattice and DOS drawn
    DOS_ax : axis instance
        axis for the DOS plot
    ax : axis instance
        axis for the time domain or other plots
    """
    fig = plt.figure(figsize=(1.5 * 5, 1.5 * 7))
    # axes constructor axes([left, bottom, width, height])
    ax = plt.axes(ax_pos)
    return fig, ax


def initialize_landscape(ax_pos=[0.1, 0.05, 0.8, 0.54]):
    """

    Parameters
    ----------
    ax_pos : list of 4 floats
        xbottom ybottom, width and height of colorbar in figure_fraction coordinates

    Returns
    ----------
    fig : matplotlib figure
        figure with lattice and DOS drawn
    ax : axis instance
        axis for the main plots
    """
    fig = plt.figure(figsize=(16. * 7.6 / 16., 9. * 7.6 / 16.))
    # axes constructor axes([left, bottom, width, height])
    ax = plt.axes(ax_pos)

    return fig, ax


def initialize_landscape_with_header(preset_cbar=False, ax_pos=[0.1, 0.05, 0.8, 0.54],
                                     cbar_pos=[0.79, 0.80, 0.012, 0.15],
                                     ax_cbar_pos=[0.9, 0.4, 0.012, 0.], ax_cbar=False):
    """

    Parameters
    ----------
    preset_cbar : bool
        If True, specifies colorbar position on plot using cbar_pos keywork argument to place the colorbar.
    cbar_pos : list of 4 floats
        xbottom ybottom, width and height of colorbar in figure_fraction coordinates

    Returns
    ----------
    fig : matplotlib figure
        figure with lattice and DOS drawn
    DOS_ax : axis instance
        axis for the DOS plot
    ax : axis instance
        axis for the time domain or other plots
    """
    fig = plt.figure(figsize=(16. * 7.6 / 16., 9. * 7.6 / 16.))
    if preset_cbar:
        header_cbar = plt.axes(cbar_pos)
        header_ax = plt.axes([0.30, cbar_pos[1], 0.45, 0.18])
    else:
        header_ax = plt.axes([0.30, cbar_pos[1], 0.50, 0.18])
        header_cbar = None
    # axes constructor axes([left, bottom, width, height])
    ax = plt.axes(ax_pos)
    if ax_cbar:
        ax_cbar = plt.axes()
    else:
        ax_cbar = None

    return fig, ax, header_ax, header_cbar, ax_cbar


def initialize_histogram(xx, alpha=1.0, colorV=None, facecolor='#80D080', nbins=75,
                         fontsize=8, linewidth=1, xlabel=None, ylabel=None, label=None):
    """draws a histogram (such as DOS plot), where each bin can be colored according to colorV.

    Parameters
    ----------
    xx : int or float array of dimension nx1
        values to histogram
    alpha: float
        Opacity value for the bars on the plot
    colorV: len(eigval) x 1 float array
        values in (0,1) to translate into colors from colormap. Values outside the range (0,1) will be ducked as 0 or 1.
    colormap: str or matplotlib.colors.Colormap instance
        The colormap to use to determine the bin colors in the histogram
    facecolor: basestring or color specification
        hexadecimal or other specification of the color of the bars on the plot.
        Only used if colorV=None, otherwise colors will be based on colormap.
    nbins : int
        The number of bins to make in the histogram
    fontsize : int
        The fontsize for the labels
    linewidth : float or int
        The width of the line outlining the histogram bins
    xlabel : str or None
        The label for the x axis
    ylabel : str or None
        The label for the y axis
    label : str or None
        The label for the legend of the histogram bins added here

    """
    fig = plt.figure(figsize=(10 * 0.6, 5 * 0.6))
    hist_ax = plt.axes([0.15, 0.25, 0.8, 0.65])  # axes constructor axes([left, bottom, width, height])
    draw_histogram(xx, hist_ax, alpha=alpha, colorV=colorV, facecolor=facecolor, nbins=nbins,
                   fontsize=fontsize, linewidth=linewidth, xlabel=xlabel, ylabel=ylabel)
    return fig, hist_ax


def draw_histogram(xx, hist_ax, alpha=1.0, colorV=None, facecolor='#80D080', edgecolor=None, nbins=75,
                   fontsize=8, linewidth=1, xlabel=None, ylabel=None, label=None):
    """draws histogram (such as DOS plot), where each bin can be colored according to colorV.

    Parameters
    ----------
    xx : int or float array of dimension nx1
        values to histogram
    hist_ax: matplotlib axis instance
        axis on which to draw the histogram
    alpha: float
        Opacity value for the bars on the plot
    colorV: len(eigval) x 1 float array
        values in (0,1) to translate into colors from colormap. Values outside the range (0,1) will be ducked as 0 or 1.
    colormap: str or matplotlib.colors.Colormap instance
        The colormap to use to determine the bin colors in the histogram
    facecolor: basestring or color specification
        hexadecimal or other specification of the color of the bars on the plot.
        Only used if colorV=None, otherwise colors will be based on colormap.
    nbins : int
        The number of bins to make in the histogram
    fontsize : int
        The fontsize for the labels
    linewidth : float or int
        The width of the line outlining the histogram bins
    xlabel : str or None
        The label for the x axis
    ylabel : str or None
        The label for the y axis
    label : str or None
        The label for the legend of the histogram bins added here
    """
    plt.sca(hist_ax)
    if colorV is None:
        n, bins, patches = hist_ax.hist(xx, nbins, histtype='stepfilled', alpha=alpha, linewidth=linewidth, label=label)
        plt.setp(patches, 'facecolor', facecolor)
        if edgecolor is not None:
            plt.setp(patches, 'edgecolor', edgecolor)
    else:
        n, bins, patches = hist_ax.hist(xx, nbins, alpha=alpha, linewidth=linewidth, label=label)

    if xlabel is not None:
        hist_ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        hist_ax.set_ylabel(ylabel, fontsize=fontsize)
    return hist_ax


def clear_plot(figure, clear_array):
    """clears plot of items specified by clear_array
    
    Parameters
    ----------
    figure : 
        figure with items to be cleared
    
    clear_array :
        items to clear from figure
    """
    for i in range(len(clear_array)):
        clear_array[i].remove()



def get_points_per_unit(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.apply_aspect()
    x0, x1 = ax.get_xlim()
    return ax.bbox.width / abs(x1 - x0)


def initialize_1panel_centered_fig(Wfig=90, Hfig=None, wsfrac=0.4, hsfrac=None, vspace=8, hspace=8,
                                   tspace=8, fontsize=8, evenpix=True, dpi=150):
    """Creates a plot with one axis in the center of the canvas.

    Parameters
    ----------
    Wfig : width of the figure in mm
    Hfig : float or None
        height of the figure in mm. If None, uses Hfig = y0 + hs + vspace + hscbar + tspace
    x0frac : fraction of Wfig to leave blank left of plot
    y0frac : fraction of Wfig to leave blank below plot
    wsfrac : fraction of Wfig to make width of subplot
    hs : height of subplot in mm. If none, uses ws = wsfrac * Wfig
    vspace : vertical space between subplots
    hspace : horizontal space btwn subplots
    tspace : space above top figure
    fontsize : size of text labels, title

    Returns
    -------
    fig
    ax
    """
    # Make figure
    x0 = Wfig * (1 - wsfrac) * 0.5
    if Hfig is None:
        Hfig = Wfig * 0.75
    ws = round(Wfig * wsfrac)
    if hsfrac is None:
        hs = ws
    else:
        hs = Wfig * hsfrac
    y0 = (Hfig - hs - tspace) * 0.5
    if evenpix:
        wpix = Wfig / 25.4 * dpi
        hpix = Hfig / 25.4 * dpi
        fig = sps.figure_in_pixels(wpix, hpix, dpi=dpi)
    else:
        fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')
    ax = sps.axes_in_mm((Wfig - ws) * 0.5, y0, ws, hs, label='', label_params=label_params)
    return fig, ax


def initialize_fullcanvas_axis(Wfig=90, Hfig=None, fig=None, units='mm', **kwargs):
    """Creates a plot with one axis covering the whole canvas.

    Parameters
    ----------
    Wfig : width of the figure in mm
    Hfig : float or None
        height of the figure in mm. If None, uses Hfig = y0 + hs + vspace + hscbar + tspace
    fig : matplotlib.pyplot.figure instance
        if figure is supplied, Wfig and Hfig are set to be the size of the supplied figure
    units : str ('mm' or 'pixels')
        Whether the supplied width/height are in mm or pixels
    **kwargs : keyword arguments for sps.axes_in_mm() or sps.axes_in_pixels() if units == 'pixels'
        keyword arguments for sps.axes_in_mm(), such as dpi=150

    Returns
    -------
    fig
    ax
    """
    # Make figure
    if fig is None:
        if Hfig is None:
            Hfig = Wfig
        if units == 'mm':
            fig = sps.figure_in_mm(Wfig, Hfig)
        else:
            fig = sps.figure_in_pixels(Wfig, Hfig, **kwargs)
    else:
        # get supplied figure size and convert to mm
        size = fig.get_size_inches() * 25.4
        Wfig, Hfig = size[0], size[1]

    # Make figure
    ax = fig.add_axes([0, 0, 1, 1], **kwargs)
    return fig, ax


def initialize_1panel_fig(Wfig=90, Hfig=None, x0frac=None, y0frac=0.1, wsfrac=0.4, hsfrac=None, vspace=8, hspace=8,
                          tspace=10, fontsize=8):
    """Creates a plot with one axis in the center of the canvas.

    Parameters
    ----------
    Wfig : width of the figure in mm
    Hfig : float or None
        height of the figure in mm. If None, uses Hfig = y0 + hs + vspace + hscbar + tspace
    x0frac : fraction of Wfig to leave blank left of plot
    y0frac : fraction of Wfig to leave blank below plot
    wsfrac : fraction of Wfig to make width of subplot
    hs : height of subplot in mm. If none, uses ws = wsfrac * Wfig
    vspace : vertical space between subplots
    hspace : horizontal space btwn subplots
    tspace : space above top figure
    fontsize : size of text labels, title

    Returns
    -------
    fig
    ax
    """
    # Make figure
    y0 = round(Wfig * y0frac)
    ws = round(Wfig * wsfrac)
    if hsfrac is None:
        hs = ws
    else:
        hs = hsfrac * Wfig
    if Hfig is None:
        Hfig = y0 + hs + tspace
    if x0frac is None:
        x0 = (Wfig - ws) * 0.5
    else:
        x0 = round(Wfig * x0frac)
    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')
    ax = sps.axes_in_mm(x0, y0, ws, hs, label='', label_params=label_params)
    return fig, ax


def initialize_1panel_cbar_fig(Wfig=90, Hfig=None, x0frac=0.15, y0frac=0.1, wsfrac=0.4, hsfrac=None,
                               wcbarfrac=0.05, hcbarfrac=0.7,
                               vspace=8, hspace=5, tspace=10, fontsize=8):
    """Creates a plot with one axis in the center of the canvas and a horizontal colorbar above it.

    Parameters
    ----------
    Wfig : width of the figure in mm
    Hfig : float or None
        height of the figure in mm. If None, uses Hfig = y0 + hs + vspace + hscbar + tspace
    x0frac : fraction of Wfig to leave blank left of plot
    y0frac : fraction of Wfig to leave blank below plot
    wsfrac : fraction of Wfig to make width of subplot
    hs : float or None
        height of subplot in mm. If None, uses ws = wsfrac * Wfig
    wcbarfrac : float
        width of the colorbar as fraction of the panel width (ws)
    hcbarfrac : float
        height of the colorbar as fraction of the panel height (hs)
    vspace : vertical space between subplots
    hspace : horizontal space btwn panel and cbar
    tspace : space above top figure
    fontsize : size of text labels, title

    Returns
    -------
    fig
    ax
    """
    # Make figure
    x0 = Wfig * x0frac
    y0 = Wfig * y0frac
    ws = Wfig * wsfrac
    if hsfrac is None:
        hs = ws
    else:
        hs = hsfrac * Wfig
    wcbar = wcbarfrac * ws
    hcbar = hcbarfrac * hs
    if Hfig is None:
        Hfig = y0 + hs + tspace

    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')
    ax = sps.axes_in_mm(x0, y0, ws, hs, label='', label_params=label_params)
    cbar_ax = sps.axes_in_mm(x0 + ws + hspace, y0 + (1. - hcbarfrac) * hs * 0.5, wcbar, hcbar,
                             label='', label_params=label_params)
    return fig, ax, cbar_ax


def initialize_1panel_centered_cbar_fig(Wfig=90, Hfig=None, y0frac=0.1, wsfrac=0.4, hsfrac=None,
                                        wcbarfrac=0.05, hcbarfrac=0.7,
                                        vspace=8, hspace=5, tspace=10, fontsize=8):
    """Creates a plot with one axis in the center of the canvas and a horizontal colorbar above it.

    Parameters
    ----------
    Wfig : width of the figure in mm
    Hfig : float or None
        height of the figure in mm. If None, uses Hfig = y0 + hs + vspace + hscbar + tspace
    x0frac : fraction of Wfig to leave blank left of plot
    y0frac : fraction of Wfig to leave blank below plot
    wsfrac : fraction of Wfig to make width of subplot
    hs : float or None
        height of subplot in mm. If None, uses ws = wsfrac * Wfig
    wcbarfrac : float
        width of the colorbar as fraction of the panel width (ws)
    hcbarfrac : float
        height of the colorbar as fraction of the panel height (hs)
    vspace : vertical space between subplots
    hspace : horizontal space btwn panel and cbar
    tspace : space above top figure
    fontsize : size of text labels, title

    Returns
    -------
    fig
    ax
    """
    # Make figure
    y0 = Wfig * y0frac
    ws = Wfig * wsfrac
    if hsfrac is None:
        hs = ws
    else:
        hs = hsfrac * Wfig
    wcbar = wcbarfrac * ws
    hcbar = hcbarfrac * hs
    if Hfig is None:
        Hfig = y0 + hs + tspace

    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')
    ax = sps.axes_in_mm((Wfig - ws) * 0.5, y0, ws, hs, label='', label_params=label_params)
    cbar_ax = sps.axes_in_mm((Wfig + ws) * 0.5 + hspace, y0 + (1. - hcbarfrac) * hs * 0.5, wcbar, hcbar,
                             label='', label_params=label_params)
    return fig, ax, cbar_ax


def initialize_nxmpanel_cbar_fig(nn, mm, Wfig=90., Hfig=None, x0frac=0.15, y0frac=0.1, wsfrac=0.2, hsfrac=None,
                                 wcbarfrac=0.05, hcbarfrac=0.7,
                                 vspace=8, hspace=5, tspace=10, fontsize=8, x0cbarfrac=None, y0cbarfrac=None,
                                 orientation='vertical', cbar_placement='default'):
    """Creates a plot with N x M axes grid (in horizontal row) and a colorbar.

    Parameters
    ----------
    nn : int
        Number of rows of axes
    mm : int
        Number of cols of axes
    Wfig : float or int
        width of the figure in mm
    Hfig : float or None
        height of the figure in mm. If None, uses Hfig = y0 + hs + vspace + hscbar + tspace
    x0frac : float
        fraction of Wfig to leave blank left of plot
    y0frac : float
        fraction of Wfig to leave blank below plot
    wsfrac : float
        fraction of Wfig to make width of subplot
    hsfrac : float or None
        height of subplot in fraction of figure width. If None, uses hs = wsfrac * Wfig
    wcbarfrac : float
        width of the colorbar as fraction of the panel width (ws)
    hcbarfrac : float
        height of the colorbar as fraction of the panel height (hs)
    vspace : float or int
        vertical space between subplots
    hspace : float or int
        horizontal space btwn panel and cbar
    tspace : float or int
        space above top figure in mm
    fontsize : int
        size of text labels, title
    x0cbarfrac : float or None
        Left position of the colorbar in units of Wfig
    y0cbarfrac : float or None
        Bottom position of the colorbar in units of Hfig
    orientation : str
    cbar_placement : str (default='default')
        Description for placement for cbar x0 and y0, used if x0cbarfrac and/or y0cbarfrac is None
        ['above_center', 'right_right', 'above_right']

    Returns
    -------
    fig
    ax
    """
    # Make figure
    x0 = Wfig * x0frac
    y0 = Wfig * y0frac
    ws = Wfig * wsfrac
    if hsfrac is None:
        hs = ws
    else:
        hs = hsfrac * Wfig
    wcbar = wcbarfrac * ws
    hcbar = hcbarfrac * hs
    if Hfig is None:
        Hfig = y0 + (hs * nn) + vspace * (nn - 1) + tspace

    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')
    ax = []
    for nii in range(nn):
        for mii in range(mm):
            ax.append(sps.axes_in_mm(x0 + (ws + hspace) * mii,
                                     y0 + (nn - nii - 1) * (hs + vspace), ws, hs, label='', label_params=label_params))

    # print 'plots at ', x0frac + wsfrac * 0.5, x0frac + 1.5 * wsfrac + hspace / Wfig, \
    #       x0frac + 2.5 * wsfrac + 2. * hspace / Wfig
    # print 'plots at ', x0 + ws * 0.5, x0 + 1.5 * ws + hspace, x0 + 2.5 * ws + 2. * hspace
    # print 'plots at ', (x0 + ws * 0.5) / Wfig, (x0 + 1.5 * ws + hspace) / Wfig, \
    #       (x0 + 2.5 * ws + 2. * hspace) / Wfig
    # Placement of the colorbar.
    # Note: to put colorbar over right subplot:
    # set x0cbarfrac = (x0frac + (mm - 0.5) * (wsfrac + hspace / Wfig) - wcbarfrac * 0.5)
    if x0cbarfrac is None:
        if orientation == 'vertical':
            if cbar_placement in ['right_right', 'default']:
                x0cbar = x0 + mm * ws + hspace * mm
            else:
                # todo: add cases here
                x0cbar = x0 + mm * ws + hspace * mm
        elif orientation == 'horizontal':
            if cbar_placement in ['above_center']:
                x0cbar = (Wfig - wcbar) * 0.5
            elif cbar_placement in ['above_right', 'default']:
                print('mm = ', mm)
                x0cbar = x0 + (mm - 0.5) * ws - wcbar * 0.5 + (mm - 1) * hspace
    else:
        x0cbar = x0cbarfrac * Wfig

    print('x0cbarfrac = ', x0cbarfrac)

    print('leplt.nxm: orientation = ', orientation, '\ncbar_placement= ', cbar_placement)
    if y0cbarfrac is None:
        if orientation == 'vertical':
            if cbar_placement in ['right_right', 'default']:
                y0cbar = y0 + (1. - hcbarfrac) * hs * 0.5
            else:
                y0cbar = y0 + (1. - hcbarfrac) * hs * 0.5
        elif orientation == 'horizontal':
            if cbar_placement in ['above_right', 'default']:
                print('nn = ', nn)
                y0cbar = y0 + (nn - 1.) * vspace + (nn + 0.1) * hs
            elif cbar_placement in ['above_center']:
                print('leplt: placing cbar in center above subplots...')
                y0cbar = y0 + (nn - 1.) * vspace + (nn + 0.1) * hs
            else:
                y0cbar = y0 + (nn - 1.) * vspace + (nn + 0.1) * hs
    else:
        y0cbar = y0cbarfrac * Hfig
    cbar_ax = sps.axes_in_mm(x0cbar, y0cbar, wcbar, hcbar, label='', label_params=label_params)

    return fig, ax, cbar_ax


def initialize_nxmpanel_fig(nn, mm, Wfig=90, Hfig=None, x0frac=0.15, y0frac=0.1, wsfrac=0.2, hsfrac=None,
                            vspace=8, hspace=5, tspace=10, fontsize=8, fig=None, panels4o3=False):
    """Creates a plot with N x M axes grid (in horizontal row)

    Parameters
    ----------
    Wfig : width of the figure in mm
    Hfig : float or None
        height of the figure in mm. If None, uses Hfig = y0 + hs + vspace + hscbar + tspace
    x0frac : fraction of Wfig to leave blank left of plot
    y0frac : fraction of Wfig to leave blank below plot
    wsfrac : fraction of Wfig to make width of subplot
    hsfrac : float or None
        height of subplot in units of Wfig. If None, uses hs = ws = wsfrac * Wfig
    vspace : vertical space between subplots
    hspace : horizontal space btwn panel and cbar
    tspace : space above top figure
    fontsize : size of text labels, title

    Returns
    -------
    fig
    ax
    """
    # Make figure
    if fig is None:
        x0, y0, ws = Wfig * x0frac, Wfig * y0frac, Wfig * wsfrac
        if hsfrac is not None:
            hs = hsfrac * Wfig
        elif panels4o3:
            hs = ws * 4. / 3.
        else:
            hs = ws
        # wcbar = wcbarfrac * ws
        # hcbar = hcbarfrac * hs
        if Hfig is None:
            Hfig = y0 + hs * nn + vspace * (nn - 1) + tspace
        fig = sps.figure_in_mm(Wfig, Hfig)
    else:
        # get supplied figure size and convert to mm
        size = fig.get_size_inches() * 25.4
        Wfig, Hfig = size[0], size[1]
        x0, y0, ws = Wfig * x0frac, Wfig * y0frac, Wfig * wsfrac
        if hsfrac is not None:
            hs = hsfrac * Wfig
        elif panels4o3:
            hs = ws * 4. / 3.
        else:
            hs = ws
    label_params = dict(size=fontsize, fontweight='normal')
    ax = []
    for nii in range(nn):
        for mii in range(mm):
            ax.append(sps.axes_in_mm(x0 + (ws + hspace) * mii,
                                     y0 + (nn - nii - 1) * (hs + vspace), ws, hs, label='', label_params=label_params))

    return fig, ax


def initialize_1panel_cbar_cent(Wfig=90, Hfig=None, wsfrac=0.4, hsfrac=None, cbar_pos='above',
                                wcbarfrac=0.6, hcbarfrac=0.05, cbar_label='',
                                vspace=8, hspace=5, tspace=10, fontsize=8, evenpix=True, dpi=150):
    """Creates a plot with one axis instance in the center of the canvas, with colorbar above the axis.

    Parameters
    ----------
    Wfig : width of the figure in mm
    Hfig : float or None
        height of the figure in mm. If None, uses Hfig = y0 + hs + vspace + hscbar + tspace
    wsfrac : float
        fraction of Wfig to make width of subplot
    hsfrac : float or None
        fraction of Wfig to make height of subplot. If None, uses hs = wsfrac * Wfig
    cbar_pos : str specifier ('above', 'right')
        Where to place the colorbar
    wcbarfrac : float
        width of the colorbar as fraction of the panel width (ws)
    hcbarfrac : float
        height of the colorbar as fraction of the panel height (hs)
    vspace : vertical space between subplots
    hspace : horizontal space btwn panel and cbar
    tspace : space above top figure
    fontsize : size of text labels, title

    Returns
    -------
    fig
    ax
    """
    # Make figure
    if Hfig is None:
        Hfig = Wfig * 0.75
    ws = round(Wfig * wsfrac)
    if hsfrac is None:
        hs = ws
    else:
        hs = Wfig * hsfrac
    y0 = (Hfig - hs - tspace) * 0.5
    wcbar = wcbarfrac * ws
    hcbar = hcbarfrac * hs
    if evenpix:
        wpix = Wfig / 25.4 * dpi
        hpix = Hfig / 25.4 * dpi
        fig = sps.figure_in_pixels(wpix, hpix, dpi=dpi)
    else:
        fig = sps.figure_in_mm(Wfig, Hfig)

    label_params = dict(size=fontsize, fontweight='normal')
    ax = sps.axes_in_mm((Wfig - ws) * 0.5, y0, ws, hs, label='', label_params=label_params)
    if cbar_pos == 'right':
        cbar_ax = sps.axes_in_mm((Wfig + ws) * 0.5 + hspace, y0 + (1. - hcbarfrac) * hs * 0.5, wcbar, hcbar,
                                 label=cbar_label, label_params=label_params)
    elif cbar_pos == 'above':
        cbar_ax = sps.axes_in_mm((Wfig - wcbar) * 0.5, y0 + hs + vspace, wcbar, hcbar,
                                 label=cbar_label, label_params=label_params)
    return fig, ax, cbar_ax


def initialize_2panel_3o4ar_cent(Wfig=360, Hfig=270, fontsize=8, wsfrac=0.4, wssfrac=0.3, x0frac=0.1, y0frac=0.1,
                                 fig=None):
    """Returns 2 panel figure with left axis square and right axis 3/4 aspect ratio

    Returns
    -------
    fig :
    ax :
    """
    if fig is None:
        fig = sps.figure_in_mm(Wfig, Hfig)
    else:
        # get supplied figure size and convert to mm
        size = fig.get_size_inches() * 25.4
        Wfig, Hfig = size[0], size[1]

    ws = wsfrac * Wfig
    hs = ws
    wss = wssfrac * Wfig
    hss = wss * 3. / 4.
    x0 = x0frac * Wfig
    y0 = y0frac * Wfig
    label_params = dict(size=fontsize, fontweight='normal')
    ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
          for x0, y0, width, height, part in (
              [x0, (Hfig - hs) * 0.5, ws, hs, ''],  # network and kitaev regions
              [Wfig - wss - x0, (Hfig - hss) * 0.5, wss, hss, '']  # plot for chern
          )]
    return fig, ax


def initialize_2panel_4o3ar_cent(Wfig=360, Hfig=270, fontsize=8, wsfrac=0.4, wssfrac=0.3, x0frac=0.1, fig=None):
    """Returns 2 panel figure with left axis square and right axis 4/3 aspect ratio

    Parameters
    ----------
    Wfig : int or float
        the width of the figure in mm
    Hfig : int or float
        the height of the figure in mm
    fontsize : int
        the font size for labels, ticks, etc
    wsfrac : float
        the fraction of Wfig to make the left (1x1 ar) panel
    wssfrac : float
        the fraction of Wfig to make the right (4x3 ar) panel
    x0frac : float
        fraction of Wfig for spacing on left and right (ie determines space between the two panels so that centered)

    Returns
    -------
    fig :
    ax :
    """
    if fig is None:
        fig = sps.figure_in_mm(Wfig, Hfig)
    else:
        # get supplied figure size and convert to mm
        size = fig.get_size_inches() * 25.4
        Wfig, Hfig = size[0], size[1]

    ws = wsfrac * Wfig
    hs = ws
    wss = wssfrac * Wfig
    hss = wss * 4. / 3.
    x0 = x0frac * Wfig
    label_params = dict(size=fontsize, fontweight='normal')
    ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
          for x0, y0, width, height, part in (
              [x0, (Hfig - hs) * 0.5, ws, hs, ''],  # network and kitaev regions
              [Wfig - wss - x0, (Hfig - hss) * 0.5, wss, hss, '']  # plot for chern
          )]
    return fig, ax


def initialize_2panel_cbar_cent(Wfig=360, Hfig=270, fontsize=12, wsfrac=0.4, wssfrac=0.3, x0frac=0.1, y0frac=0.1,
                                wcbarfrac=0.15, hcbar_fracw=0.1, vspace=5, right3o4=False, right4o3=False):
    """Returns 2 panel figure with left axis square and right axis either square or 3/4 aspect ratio, depending on
     if right3o4 is True,  but also 2 colorbar axes (one for each panel).
     The colorbars are above the plot axes by default.

    Parameters
    ----------
    Wfig : int or float
        the width of the figure in mm
    Hfig : int or float
        the height of the figure in mm
    fontsize : int
        the font size for labels, ticks, etc
    wsfrac : float
        the fraction of Wfig to make the left (1x1 ar) panel
    wssfrac : float
        the fraction of Wfig to make the right (4x3, 3x4, or 1x1 ar) panel, depending on whether right3o4/4o3 are True
    x0frac : float
        fraction of Wfig for spacing on left and right (ie determines space between the two panels so that centered)
    hcbar_fracw : float
        height of the colorbar, as a fraction of the width of the colorbar

    Returns
    -------
    fig :
    ax :
    cbar_ax:
    """
    fig = sps.figure_in_mm(Wfig, Hfig)
    ws = wsfrac * Wfig
    hs = ws
    wss = wssfrac * Wfig
    if right3o4:
        hss = wss * 3. / 4.
    elif right4o3:
        hss = wss * 4. / 3.
    else:
        hss = wss
    wcbar = wcbarfrac * Wfig
    hcbar = hcbar_fracw * wcbar
    x0 = x0frac * Wfig
    y0 = y0frac * Wfig
    label_params = dict(size=fontsize, fontweight='normal')
    ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
          for x0, y0, width, height, part in (
              [x0, (Hfig - hs) * 0.5, ws, hs, ''],  # network and kitaev regions
              [Wfig - wss - x0, (Hfig - hss) * 0.5, wss, hss, '']  # plot for chern
          )]
    cbar_ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
               for x0, y0, width, height, part in (
                   [x0 + (hs - wcbar) * 0.5, (Hfig + hs) * 0.5 + vspace, wcbar, hcbar, ''],  # left cbar above
                   [Wfig - wss - x0 + (hs - wcbar) * 0.5,
                    (Hfig + hs) * 0.5 + vspace, wcbar, hcbar, '']  # right cbar
               )]
    for cax in cbar_ax:
        print('cax.get_position() = ', cax.get_position())

    return fig, ax, cbar_ax


def initialize_2panel_centy(Wfig=360, Hfig=270, fontsize=12, wsfrac=0.4, wssfrac=0.3, x0frac=0.1, hspace=10,
                            right3o4=False, right4o3=False):
    """Returns 2 panel figure with left axis square and right axis either square or 3/4 aspect ratio, depending on
     if right3o4 is True,  but also 1 colorbar axes (one for the right panel).
     The colorbar can be placed anywhere.

    Parameters
    ----------

    Wfig : int or float
        the width of the figure in mm
    Hfig : int or float
        the height of the figure in mm
    fontsize : int
        the font size for labels, ticks, etc
    wsfrac : float
        the fraction of Wfig to make the left (1x1 ar) panel
    wssfrac : float
        the fraction of Wfig to make the right (4x3 ar) panel
    x0frac : float
        fraction of Wfig for spacing on left and right (ie determines space between the two panels so that centered)
    wcbarfrac: float
        width of the colorbar, as a fraction of the width of the axis
    hcbar_fracw : float
        height of the colorbar, as a fraction of the width of the colorbar

    Returns
    -------
    fig :
    ax :
    cbar_ax:
    """
    fig = sps.figure_in_mm(Wfig, Hfig)
    ws = wsfrac * Wfig
    hs = ws
    wss = wssfrac * Wfig
    if right3o4:
        hss = wss * 3. / 4.
    elif right4o3:
        hss = wss * 4. / 3.
    else:
        hss = wss
    x0 = x0frac * Wfig
    label_params = dict(size=fontsize, fontweight='normal')
    ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
          for x0, y0, width, height, part in (
              [x0, (Hfig - hs) * 0.5, ws, hs, ''],
              [x0 + ws + hspace, (Hfig - hss) * 0.5, wss, hss, '']  # right panel
          )]
    return fig, ax


def initialize_2panel_1cbar_centy(Wfig=360, Hfig=270, fontsize=12, wsfrac=0.4, wssfrac=0.3,
                                  hsfrac=None, hssfrac=None, x0frac=0.1, hspace=10,
                                  wcbarfrac=0.15, hcbar_fracw=0.1, x0cbar_frac=0.9, y0cbar_frac=0.3,
                                  right3o4=False, right4o3=False):
    """Returns 2 panel figure with left axis square and right axis either square or 3/4 aspect ratio, depending on
     if right3o4 is True,  but also 1 colorbar axes (one for the right panel).
     The colorbar can be placed anywhere.

    Parameters
    ----------

    Wfig : int or float
        the width of the figure in mm
    Hfig : int or float
        the height of the figure in mm
    fontsize : int
        the font size for labels, ticks, etc
    wsfrac : float
        the fraction of Wfig to make the left (1x1 ar) panel
    wssfrac : float
        the fraction of Wfig to make the right (4x3 ar) panel
    x0frac : float
        fraction of Wfig for spacing on left and right (ie determines space between the two panels so that centered)
    wcbarfrac: float
        width of the colorbar, as a fraction of the width of the axis
    hcbar_fracw : float
        height of the colorbar, as a fraction of the width of the colorbar

    Returns
    -------
    fig :
    ax :
    cbar_ax:
    """
    fig = sps.figure_in_mm(Wfig, Hfig)
    ws = wsfrac * Wfig
    if hsfrac is not None:
        hs = hsfrac * Wfig
    else:
        hs = ws

    wss = wssfrac * Wfig
    if right3o4:
        hss = wss * 3. / 4.
    elif hssfrac is not None:
        hss = hssfrac * Wfig
    else:
        hss = wss

    wcbar = wcbarfrac * Wfig
    hcbar = hcbar_fracw * wcbar
    x0 = x0frac * Wfig
    x0cbar = x0cbar_frac * Wfig
    y0cbar = y0cbar_frac * Hfig
    label_params = dict(size=fontsize, fontweight='normal')
    ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
          for x0, y0, width, height, part in (
              [x0, (Hfig - hs) * 0.5, ws, hs, ''],
              [x0 + ws + hspace, (Hfig - hss) * 0.5, wss, hss, '']  # right panel
          )]
    cbar_ax = sps.axes_in_mm(x0cbar, y0cbar, wcbar, hcbar, label='', label_params=label_params)

    return fig, ax, cbar_ax


def initialize_axis_stack(n_ax, make_cbar=False, Wfig=90, Hfig=90, hfrac=None, wfrac=0.6, x0frac=None, y0frac=0.12,
                          vspace=5, hspace=5, fontsize=8, wcbar_frac=0.2, cbar_aratio=0.1, cbar_orientation='vertical',
                          cbarspace=5, tspace=8, **kwargs):
    """Create a vertical stack of plots, and a colorbar if make_cbar is True

    Parameters
    ----------
    n_ax : int
        number of axes to draw
    make_cbar : bool
        Create a colorbar on the figure
    Wfig : int or float
        width of figure in mm
    Hfig : int or float
        height of figure in mm
    hfrac : float or None
        Fraction of Hfig to make each axis height (hs)
    wfrac : float
        Fraction of Wfig to make each axis width (ws)
    x0frac : float or None
        Buffer room in mm on the left of all the axes in the stack. If None, centers the axes.
    y0frac : float or None
        Buffer room in mm on the bottom of the lowest axis in the stack
    vspace : float or int
        vertical space in mm between each axis in the stack
    hspace : float or int
        space between the stack of axes and the colorbar, in mm
    fontsize : int
        font size for axis params
    hcbar_frac : float
        fraction of Wfig to make height of colorbar
    cbar_orientation : str ('vertical', 'horizontal')
        Orientation of the colorbar

    """
    # This method returns and ImageGrid instance
    # ax = AxesGrid(fig, 111,  # similar to subplot(111)
    #               nrows_ncols=(n_ax, 1),  # creates 2x2 grid of axes
    #               axes_pad=0.1,  # pad between axes in inch.
    #               share_all=True,
    #               )
    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')

    if hfrac is None:
        hfrac = 0.8 / float(n_ax) - ((n_ax - 2.) * float(vspace) + tspace) / (float(Hfig) * float(n_ax))
        if make_cbar and cbar_orientation == 'horizontal':
            # colorbar is going on top, with space cbarspace
            hfrac -= float(cbarspace) / (float(Hfig) * float(n_ax))
        print('hfrac = ', hfrac)
    if x0frac is None:
        x0 = (1. - wfrac) * 0.5 * Wfig
    else:
        x0 = x0frac * Wfig
    y0 = y0frac * Hfig
    ws = wfrac * Wfig
    hs = hfrac * Hfig
    print('hs = ', hs)
    xywh_list = [[x0, y0 + (n_ax - 1 - ii) * (hs + vspace), ws, hs, None] for ii in range(n_ax)]

    print('xywh_list = ', xywh_list)
    ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params, **kwargs)
          for x0, y0, width, height, part in xywh_list]

    if make_cbar:
        wcbar = Wfig * wcbar_frac
        hcbar = cbar_aratio * wcbar
        if cbar_orientation == 'vertical':
            cbar_ax = sps.axes_in_mm(x0 + ws + hspace, (Hfig - wcbar) * 0.5, hcbar, wcbar, label='',
                                     label_params=label_params, **kwargs)
        elif cbar_orientation == 'horizontal':
            cbar_ax = sps.axes_in_mm(x0 + (ws - wcbar) * 0.5, y0 + n_ax * (hs + vspace) + cbarspace, wcbar, hcbar,
                                     label='', label_params=label_params)
    else:
        cbar_ax = None

    return fig, ax, cbar_ax


def initialize_axis_doublestack(n_ax, make_cbar=False, Wfig=90, Hfig=90, hfrac=None, wfrac=0.3, x0frac=None,
                                y0frac=0.12, vspace=5, hspace=5, fontsize=8, wcbar_frac=0.2, cbar_aratio=0.1,
                                cbarspace=5, tspace=8):
    """Create a vertical stack of plots, and a colorbar if make_cbar is True

    Parameters
    ----------
    n_ax : int
        number of axes to draw
    make_cbar : bool
        Create a colorbar on the figure (will be horizontal, above subplots)
    Wfig : int or float
        width of figure in mm
    Hfig : int or float
        height of figure in mm
    hfrac : float or None
        Fraction of Hfig to make each axis height (hs)
    wfrac : float
        Fraction of Wfig to make each axis width (ws)
    x0frac : float or None
        Buffer room in mm on the left of all the axes in the stack. If None, centers the axes.
    y0frac : float or None
        Buffer room in mm on the bottom of the lowest axis in the stack
    vspace : float or int
        vertical space in mm between each axis in the stack
    hspace : float or int
        space between the stack of axes and the colorbar, in mm
    fontsize : int
        font size for axis params
    hcbar_frac : float
        fraction of Wfig to make height of colorbar

    Returns
    -------
    fig : matplotlib figure instance
    ax : list of matplotlib axis instances
    cbar_ax : list of 2 matplotlib axis instances

    """
    # This method returns and ImageGrid instance
    # ax = AxesGrid(fig, 111,  # similar to subplot(111)
    #               nrows_ncols=(n_ax, 1),  # creates 2x2 grid of axes
    #               axes_pad=0.1,  # pad between axes in inch.
    #               share_all=True,
    #               )
    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')

    if hfrac is None:
        hfrac = 0.8 / float(n_ax) - ((n_ax - 2.) * float(vspace) + tspace) / (float(Hfig) * float(n_ax))
        if make_cbar:
            # colorbar is going on top, with space cbarspace
            hfrac -= float(cbarspace) / (float(Hfig) * float(n_ax))

    if x0frac is None:
        x0 = (1. - 2. * wfrac - float(hspace) / float(Wfig)) * 0.5 * Wfig
    else:
        x0 = x0frac * Wfig

    y0 = y0frac * Hfig
    ws = wfrac * Wfig
    hs = hfrac * Hfig
    xywh_list = [[x0, y0 + (n_ax - 1 - ii) * (hs + vspace), ws, hs, ''] for ii in range(n_ax)]
    xywh2_list = [[x0 + ws + hspace, y0 + (n_ax - 1 - ii) * (hs + vspace), ws, hs, ''] for ii in range(n_ax)]

    ax = [sps.axes_in_mm(x, y, width, height, label=part, label_params=label_params)
          for x, y, width, height, part in xywh_list]
    ax += [sps.axes_in_mm(x, y, width, height, label=part, label_params=label_params)
           for x, y, width, height, part in xywh2_list]
    if make_cbar:
        wcbar = Wfig * wcbar_frac
        hcbar = cbar_aratio * wcbar
        cbar_ax = [sps.axes_in_mm(x0 + (ws - wcbar) * 0.5, y0 + n_ax * (hs + vspace) + cbarspace, wcbar,
                                  hcbar, label='', label_params=label_params),
                   sps.axes_in_mm(x0 + (3. * ws - wcbar) * 0.5 + hspace, y0 + n_ax * (hs + vspace) + cbarspace, wcbar,
                                  hcbar, label='', label_params=label_params)]
    else:
        cbar_ax = None

    return fig, ax, cbar_ax


def initialize_insetaxis_doublestack(n_ax, make_cbar=False, Wfig=90, Hfig=90, hfrac=None, wfrac=0.3, x0frac=None,
                                     y0frac=0.12, vspace=5, hspace=5, fontsize=8, wcbar_frac=0.2, cbar_aratio=0.1,
                                     cbarspace=5, tspace=8, ins_pad=3, ins_pad_right=None, wins=None, hins=None):
    """Create a vertical stack of plots, and a colorbar if make_cbar is True

    Parameters
    ----------
    n_ax : int
        number of axes to draw
    make_cbar : bool
        Create a colorbar on the figure (will be horizontal, above subplots)
    Wfig : int or float
        width of figure in mm
    Hfig : int or float
        height of figure in mm
    hfrac : float or None
        Fraction of Hfig to make each axis height (hs)
    wfrac : float
        Fraction of Wfig to make each axis width (ws)
    x0frac : float or None
        Buffer room in mm on the left of all the axes in the stack. If None, centers the axes.
    y0frac : float or None
        Buffer room in mm on the bottom of the lowest axis in the stack
    vspace : float or int
        vertical space in mm between each axis in the stack
    hspace : float or int
        space between the stack of axes and the colorbar, in mm
    fontsize : int
        font size for axis params
    hcbar_frac : float
        fraction of Wfig to make height of colorbar

    Returns
    -------
    fig : matplotlib figure instance
    ax : list of matplotlib axis instances
    cbar_ax : list of 2 matplotlib axis instances

    """
    # This method returns and ImageGrid instance
    # ax = AxesGrid(fig, 111,  # similar to subplot(111)
    #               nrows_ncols=(n_ax, 1),  # creates 2x2 grid of axes
    #               axes_pad=0.1,  # pad between axes in inch.
    #               share_all=True,
    #               )
    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')

    if hfrac is None:
        hfrac = 0.8 / float(n_ax) - ((n_ax - 2.) * float(vspace) + tspace) / (float(Hfig) * float(n_ax))
        if make_cbar:
            # colorbar is going on top, with space cbarspace
            hfrac -= float(cbarspace) / (float(Hfig) * float(n_ax))

    if x0frac is None:
        x0 = (1. - 2. * wfrac - float(hspace) / float(Wfig)) * 0.5 * Wfig
    else:
        x0 = x0frac * Wfig

    y0 = y0frac * Hfig
    ws = wfrac * Wfig
    hs = hfrac * Hfig
    xywh_list = [[x0, y0 + (n_ax - 1 - ii) * (hs + vspace), ws, hs, ''] for ii in range(n_ax)]
    xywh2_list = [[x0 + ws + hspace, y0 + (n_ax - 1 - ii) * (hs + vspace), ws, hs, ''] for ii in range(n_ax)]

    ax = [sps.axes_in_mm(x, y, width, height, label=part, label_params=label_params)
          for x, y, width, height, part in xywh_list]
    ax += [sps.axes_in_mm(x, y, width, height, label=part, label_params=label_params)
           for x, y, width, height, part in xywh2_list]
    if make_cbar:
        wcbar = Wfig * wcbar_frac
        hcbar = cbar_aratio * wcbar
        cbar_ax = [sps.axes_in_mm(x0 + (ws - wcbar) * 0.5, y0 + n_ax * (hs + vspace) + cbarspace, wcbar,
                                  hcbar, label='', label_params=label_params),
                   sps.axes_in_mm(x0 + (3. * ws - wcbar) * 0.5 + hspace, y0 + n_ax * (hs + vspace) + cbarspace, wcbar,
                                  hcbar, label='', label_params=label_params)]
    else:
        cbar_ax = None

    if wins is None:
        wins = hs
    if hins is None:
        hins = hs
    if ins_pad_right is None:
        ins_pad_right = ins_pad
    x0ins = min(max(x0 - wins - ins_pad, 0), x0 * 0.5)
    xywh_list = [[x0ins, y0 + (n_ax - 1 - ii) * (hs + vspace) + (hs - hins) * 0.5, wins, hins, ''] for ii in
                 range(n_ax)]
    xywh2_list = [[x0 + ws * 2 + hspace + ins_pad_right, y0 + (n_ax - 1 - ii) * (hs + vspace) + (hs - hins) * 0.5,
                   wins, hins, ''] for ii in range(n_ax)]
    inset_ax = [sps.axes_in_mm(x, y, width, height, label=part, label_params=label_params)
                for x, y, width, height, part in xywh_list]
    inset_ax += [sps.axes_in_mm(x, y, width, height, label=part, label_params=label_params)
                 for x, y, width, height, part in xywh2_list]
    return fig, ax, cbar_ax, inset_ax


def cbar_ax_is_vertical(cbar_ax):
    """Determine if a colorbar axis is vertical or not (ie it is horizontal) based on its dimensions"""
    bbox = cbar_ax.get_window_extent()  # .transformed(fig.dpi_scale_trans.inverted())
    return bbox.width < bbox.height


