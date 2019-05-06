import matplotlib.pyplot as plt
import subprocess
import numpy as np
import basics.dataio as dio
import basics.stringformat as sf
import basics.plotting.plotting as leplt
import glob


def make_movie(imgname, movname, indexsz='05', framerate=10, imgdir=None, rm_images=False, save_into_subdir=False):
    """Create a movie from a sequence of images. Options allow for deleting folder automatically after making movie.
    Will run './ffmpeg', '-framerate', str(int(framerate)), '-i', imgname + '%' + indexsz + 'd.png', movname + '.mov',
         '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    Parameters
    ----------
    imgname : str
        path and filename for the images to turn into a movie
    movname : str
        path and filename for output movie
    indexsz : str
        string specifier for the number of indices at the end of each image (ie 'file_000.png' would merit '03')
    framerate : int (float may be allowed)
        The frame rate at which to write the movie
    imgdir : str or None
        name of subdirectory to delete if rm_images and save_into_subdir are both True, ie folder containing the images
        Note: this is not the full path if save_into_subir is False.
    rm_images : bool
        Remove the images from disk after writing to movie
    save_into_subdir : bool
        The images are saved into a folder which can be deleted after writing to a movie, if rm_images is True and
        imgdir is not None (ie images are not on same heirarchical level as movie or other data)
    """
    # Convert indexsz to a string if not already one
    if isinstance(indexsz, int):
        indexsz = str(indexsz)
    elif isinstance(indexsz, float):
        indexsz = str(int(indexsz))

    if movname[-4:] != '.mov':
        movname += '.mov'

    call_list = ['./ffmpeg', '-framerate', str(int(framerate)), '-i', imgname + '%' + indexsz + 'd.png',
                 movname, '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0',
                 '-r', '100', '-pix_fmt', 'yuv420p']
    print('calling: ', call_list)
    subprocess.call(call_list)

    # Delete the original images
    if rm_images:
        print('Deleting the original images...')
        if save_into_subdir and imgdir is not None:
            print('Deleting folder ' + imgdir)
            subprocess.call(['rm', '-r', imgdir])
        else:
            print('Deleting folder contents ' + imgname + '*.png')
            subprocess.call(['rm', imgname + '*.png'])
