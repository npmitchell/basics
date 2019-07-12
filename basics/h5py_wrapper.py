import numpy as np
import h5py
import os.path

"""
Store a dictionary into a hdf5 file
Look recursively for any dictionary contained
"""

# ############# Generating and Writing ####################


def save(filename, ddict, key=None):
    """

    Parameters
    ----------
    filename
    ddict

    Returns
    -------

    """
    if filename[-5:] != '.hdf5' and filename[-3:] != '.h5':
        filename = filename + '.hdf5'

    f, b = create(filename)
    if b:
        write_rec(f, ddict, key=key)
    else:
        print(filename + ' cannot be saved')

    return f


def create(filename, overwrite=False):
    # filename = file_architecture.os_i(filename)
    if not os.path.exists(filename):
        print(filename)
        f = h5py.File(filename, 'w')
        return f, True
    else:
        if overwrite:
            f = h5py.File(filename, 'w')
            return f, True
        else:
            print("File " + filename + " already exists, skip ")
            return None, False


def write(obj, overwrite=False, filename=None, key=''):
    """Write into a hdf5 file all the parameters recursively
    hdf5 file contains a dictionary for each Class instance (e.g. Sdata, param, id) the parameters
    each individual is a dictionary containing the attributes of the class + '__module__' and '__doc__'

    Parameters
    ----------
    obj : Class instance 
        to be writen in hdf5 file. Attribute can be any type, numpy array are replaced by List.
    overwrite : bool, default False
        Overwrite previous file/data
    filename : str or None
    key : str

    Returns
    -------
    None
    
    """
    # dict_total = get_attr_rec({},obj,[])
    dict_total = {}
    if filename is None:
        filename = os.path.dirname(obj.Sdata.fileCine) + '/hdf5/test' + '.hdf5'

    f, do = create(filename, overwrite=overwrite)
    if do:
        write_rec(f, obj, key=key)
        f.close()
    else:
        raise RuntimeError('Could not create file: ' + filename)


def write_rec(f, ddict, key='', grp=None, group=None, t=0, tmax=3):
    """Write recursively a dictionary into a h5py previously open file (f)

    Parameters
    ----------
    f :
    ddict :
    key : str
    grp : or None
    group : or None
    t : int
    tmax : int
    """
    done = False
    if type(ddict) == dict:
        # print("Write :" +str(ddict))
        done = True

        if group is None:
            group = key
            # grp = f.create_group(group)
        else:
            #            if 'object' in :
            group = group + '/' + key

        if group not in f:
            print(group)
            grp = f.create_group(group)
        else:
            grp = f[group]
            # print(ddict.keys())
        
        for key in ddict.keys():
            if t < tmax:  # limit the number of recursion to 2 : protection against overflow
                write_rec(f, ddict[key], key=key, group=group, grp=grp, t=t + 1)

    if type(ddict) in [list]:
        done = True
        ddict = np.asarray(ddict)

    if type(ddict) in [np.ndarray]:
        done = True
        if group is not None:
            dataname = group + '/' + key
        else:
            dataname = key

        if dataname not in f:
            dset = f.create_dataset(dataname, data=ddict, chunks=True)  # ddict.shape, dtype=ddict.dtype)
        else:
            f[dataname][...] = ddict  # dimensions should already match !!!

    if type(ddict) in [bool, int, str, float, np.int64, np.float64]:
        done = True
        # print(key)
        grp.attrs[key] = ddict

    if not done:
        print("Unrecognized : " + str(key) + ' of type ' + str(type(ddict)))


# ############## Open and load ###########

def h5open(filename, typ='r'):
    """
    Open a hdf5 file. 
    Partially cross platform function : can switch the directory name between linux and mac syntax
    """
    # filename = file_architecture.os_i(filename)
    if os.path.exists(filename):
        f = h5py.File(filename, typ)
    else:
        print("File " + filename + " does not exist")
        f = h5py.File(filename, 'w')
    return f

# print('done')

def load_dset(fn, dset_name):
    """

    Parameters
    ----------
    fn : str
        the HDF5 file to load a dataset from
    dset_name : str
        the name of the dataset to load in the hdf5 file fn
    
    Returns
    -------
    dset : npy array
        
    """
    f = h5py.File(fn, 'r')
    dset = f[dset_name][:]
    f.close()
    return dset


def load_dict(data, ans={}):
    """Transform a h5py group into a dictionary recursively.

    Parameters
    ----------
    data
    ans

    Returns
    -------
    ans : dict

    Examples
    --------
    """
    for key, item in data.attrs.items():
        ans[key] = item
    for key, item in data.items():
        if type(item) is h5py._hl.dataset.Dataset:
            #   print(item)
            ans[key] = item.value
            #    print(key,getattr(self,key))
        elif type(item) is h5py._hl.group.Group:
            ans[key] = load_dict(item, ans=ans)
    #                    load(item,data,path='')
    #   print(key,item)
    #    print("Contain subgroup ! iterate the generator")
    return ans


def display(ddict, key=None):
    """Recursive display of a dictionary

    Parameters
    ----------
    ddict
    key

    Returns
    -------

    """
    # recursive display of a dictionary
    if type(ddict) == dict:
        for key in ddict.keys():
            display(ddict[key], key=key)
    else:
        print('     ' + key + '  , ' + str(type(ddict)))


# ################### How to use it ! #####################

if __name__ == "__main__":
    def example(filename):
        f, b = create(filename)

        ddict = {'Data': np.zeros(10), 'param': 0}
        if b:
            write_rec(f, ddict, key='U')


    def example_2(filename, ddict):
        f, b = create(filename)  # create the file. b=True if it worked

        if b:
            write_rec(f, ddict, key='U')  # write recursively ddict into f

        f = h5open(filename)  # open an hdf5 file

        data = load_dict(f)  # load recursively f into data.
        # data has the same shape as the initial ddict

        return data
