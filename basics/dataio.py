import os
import shutil
import glob
import numpy as np


'''General use functions for input/output of data and files.
This module ONLY relies on built-in modules -> no dependencies on other lepm modules are allowed!
'''


def ensure_dir(f):
    """Check if directory exists, and make it if not.

    Parameters
    ----------
    f : string
        directory path to ensure

    Returns
    ----------
    """
    d = os.path.dirname(f)
    if not os.path.exists(d):
        print 'le.ensure_dir: creating dir: ', d
        os.makedirs(d)


def find_dir_with_name(name, searchdir):
    """Return a path or list of paths to directories which match the string 'name' (can have wildcards) in searchdir.
    Note that this function returns names with a trailing back slash (/)"""
    if name == '':
        '''no name given, no name returned'''
        return []
    else:
        possible_dirs = glob.glob(searchdir + name)
        okdirs = [os.path.isdir(possible_dir) for possible_dir in possible_dirs]
        out = [possible_dirs[i] + '/' for i in range(len(okdirs)) if okdirs[i]]
        if len(out) == 1:
            return out[0]
        else:
            return out


def get_fname_and_index_size(hostdir):
    """Obtain the name of the first nontrivial file in a directory and the length of its numerical index.
    For example, if "hostdir/xyz00001.txt" exist, get_fname_and_index_size(hostdir) returns ('xyz', 5).
    This function is robust to having '.'s in the filename.

    Parameters
    ----------
    hostdir : string
        Directory in which the files you want to grab exist.

    Returns
    ----------
    fname : string
        First file name in that directory
    index_size : int
        Length of the index (how many numbers make up its index)
    """
    filename = sorted(glob.glob(hostdir + '*.*'))[0]
    index_size = len((filename.split('_')[-1]).split('.')[0])
    exten_size = len((filename.split('_')[-1]).split('.')[1])
    fname = (filename.split('/')[-1])[0:-index_size - exten_size - 1]

    return fname, index_size


def read_program(filename):
    """Turn whole file into string (for compiling, for instance)"""
    # read in the OpenCL source file as a string
    f = open(filename, 'r')
    fstr = "".join(f.readlines())
    return fstr


def prepdir(dir):
    """Make sure that the variable dir ends with the character '/'.
    This prepares the string variable 'dir' to be an output directory.

    Parameters
    ----------
    dir : string
        the directory path

    Returns
    ----------
    dir : string
        the directory path, ending with '/'
    """
    if dir[-1] == '/':
        return dir
    else:
        return dir + '/'


def find_subdirs(string, maindir):
    """Find subdir(s) matching string, in maindir. Return subdirs as list.
    If there are multiple matching subdirectories, returns list of strings.
    If there are no matches, returns empty list.

    Parameters
    ----------
    string : str
        The string to match (could have wildcard char (*))
    maindir : str
        The directory in which to search for the subdirectory.
    """
    maindir = prepdir(maindir)
    print 'dio: searching for ' + maindir + string
    contents = sorted(glob.glob(maindir + string))
    is_subdir = [os.path.isdir(ii) for ii in contents]

    if len(is_subdir) == 0:
        print 'WARNING! Found no matching subdirectory: returning empty list'
        return is_subdir
    else:
        subdirs = [prepdir(contents[ii]) for ii in np.where(is_subdir)[0].tolist()]

    return subdirs


def find_subsubdirectory(string, maindir):
    """Find subsubdir matching string, in maindir. Return subdir and subsubdir names.
    If there are multiple matching subdirectories, returns list of strings.
    If there are no matches, returns empty lists.
    """
    maindir = prepdir(maindir)
    # print 'maindir = ', maindir
    contents = glob.glob(maindir + '*')
    is_subdir = [os.path.isdir(ii) for ii in contents]

    if len(is_subdir) == 0:
        print 'WARNING! Found no matching subdirectory: returning empty list'
        return is_subdir, is_subdir
    else:
        # print 'contents = ', contents
        subdirs = [contents[ii] for ii in np.where(is_subdir)[0].tolist()]
        # print 'subdirs = ', subdirs

    found = False
    subsubdir = []
    for ii in subdirs:
        # print 'ii =', ii
        # print 'prepdir(ii)+string = ',prepdir(ii)+string
        subcontents = glob.glob(prepdir(ii) + string)
        # print 'glob.glob(',prepdir(ii),string,') = ',subcontents
        is_subsubdir = [os.path.isdir(jj) for jj in subcontents]
        subsubdirs = [subcontents[jj] for jj in np.where(is_subsubdir)[0].tolist()]
        # print 'subsubdirs = ', subsubdirs
        if len(subsubdirs) > 0:
            if not found:
                if len(subsubdirs) == 1:
                    subdir = prepdir(ii)
                    subsubdir = prepdir(subsubdirs[0])
                    # print 'adding first subdir = ', subdir
                    found = True
                elif len(subsubdirs) > 1:
                    subdir = [prepdir(ii)] * len(subsubdirs)
                    # print 'adding first few subdir = ', subdir
                    found = True
                    subsubdir = [0] * len(subsubdirs)
                    for j in range(len(subsubdirs)):
                        subsubdir[j] = prepdir(subsubdirs[j])
            else:
                # Since already found one, add another
                # print ' Found more subsubdirs'
                # print 'subdir = ', subdir

                # Add subdir to list
                if isinstance(subdir, str):
                    subdir = [subdir, prepdir(ii)]
                    # print 'adding second to subdir = ', subdir
                    if len(subsubdirs) > 1:
                        for kk in range(1, len(subsubdirs)):
                            subdir.append(prepdir(ii))
                            # print 'adding second (multiple) to subdir = ', subdir
                else:
                    for kk in range(1, len(subsubdirs)):
                        subdir.append(prepdir(ii))
                        # print 'subsubdirs = ', subsubdirs
                        # print 'adding more to subdir = ', subdir
                # Add subsubdir to list
                for jj in subsubdirs:
                    if isinstance(subsubdir, str):
                        subsubdir = [subsubdir, prepdir(jj)]
                        # print 'adding second to subsubdirs = ', subsubdir
                    else:
                        subsubdir.append(prepdir(jj))
                        # print 'adding more to subsubdirs = ', subsubdir

    if found:
        return subdir, subsubdir
    else:
        return [], []


def save_dict(Pdict, filename, header, keyfmt='auto', valfmt='auto', padding_var=7):
    """Writes dictionary to txt file where each line reads 'key    : value'.

    Parameters
    ----------
    Pdict : dict
        dictionary of key, value pairs to write as txt file
    header : string
        header for the text file specifying content of the file
    keyfmt : string
        string formatting for keys, by default this is 'auto'. If not 'auto', then all keys are formatted identically.
    valfmt : string
        string formatting for value, by default this is 'auto'. If not 'auto', then all values are formatted identically.

    Returns
    ----------
    """
    with open(filename, 'w') as myfile:
        if '#' in header:
            myfile.write(header + '\n')
        else:
            myfile.write('# ' + header + '\n')

    # if keyfmt == 'auto' and valfmt == 'auto':
    for key in Pdict:
        with open(filename, 'a') as myfile:
            # print 'Writing param ', str(key)
            # print ' with value ', str(Pdict[key])
            # print ' This param is of type ', type(Pdict[key])
            if isinstance(Pdict[key], str):
                myfile.write('{{0: <{}}}'.format(padding_var).format(str(key)) + '= ' + Pdict[key] + '\n')
            elif isinstance(Pdict[key], np.ndarray):
                myfile.write('{{0: <{}}}'.format(padding_var).format(str(key)) +
                             '= ' + ", ".join(np.array_str(Pdict[key], precision=18).split()).replace('[,', '[') + '\n')
            elif isinstance(Pdict[key], list):
                myfile.write('{{0: <{}}}'.format(padding_var).format(str(key)) + '= ' + str(Pdict[key]) + '\n')
            elif isinstance(Pdict[key], tuple):
                myfile.write('{{0: <{}}}'.format(padding_var).format(str(key)) + '= ' + str(Pdict[key]) + '\n')
            elif Pdict[key] is None:
                # Don't write key to file if val is None
                pass
                # myfile.write('{{0: <{}}}'.format(padding_var).format(str(key)) + '= none' + '\n')
            else:
                # print 'dio.save_dict(): ', key, ' = ', Pdict[key]
                # print 'isstr --> ', isinstance(Pdict[key], str)
                myfile.write('{{0: <{}}}'.format(padding_var).format(str(key)) +
                             '= ' + '{0:0.18e}'.format(Pdict[key]) + '\n')


def remove(path):
    """Removes/deletes a file or folder from disk. param <path> could either be relative or absolute.

    Parameters
    ----------
    path : str
        The file or folder to delete
    """
    if os.path.isfile(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


