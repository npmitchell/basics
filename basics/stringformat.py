import numpy as np

'''Module for string format handling and transformations'''

##########################################
# String Formatting
##########################################


def is_number(s):
    """Check if a string can be represented as a number; works for floats"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def string_to_array(string, delimiter='/', dtype='auto'):
    """Convert string to numpy array
    See also string_sequence_to_numpy_array()
    """
    slist = string.split(delimiter)
    if dtype != 'auto':
        arr = np.array(slist, dtype=dtype)
    else:
        try:
            arr = np.array(slist, dtype=float)
        except:
            try:
                arr = np.array(slist, dtype=int)
            except:
                arr = np.array(slist, dtype=str)
    return arr


def string_sequence_to_numpy_array(vals, dtype=str, corrections=True):
    """Convert a string of numbers like (#/#/#) or (#:#:#) or (#:#) or (#) to a numpy array.

    Parameters
    ----------
    vals : string
        List of values, can take formats:
        'check_string_for_empty', #/#/#, #:#:#, #:#, or #
    dtype : string or 'float', 'str', 'int', etc
        the data type for the elements of the output numpy array
    corrections : bool
        Whether to make replacements of n -> -, and p -> .

    Returns
    ----------
    out : numpy array
        Values input as string, returned as array
    """
    # ensure input is string
    if not isinstance(vals, str):
        vals = str(vals)

    if corrections:
        vals = vals.replace('n', '-').replace('p', '.')

    # print 'vals = ', vals
    # sys.exit()

    # Convert to an array of strings, or to an array of dtype
    if vals == 'check_string_for_empty':
        out = np.array([])
    elif '/' not in vals and ':' not in vals:
        out = np.array([vals], dtype=dtype)
    elif ':' in vals:
        values = vals.split(':')
        if '.' in values[0] or '.' in values[1]:
            start = float(values[0])
            if len(values) == 3:
                step = float(values[1])
                end = float(values[2])
            elif len(values) == 2:
                step = float(1)
                end = float(values[1])
            else:
                raise RuntimeError('If : is used, vals must be ##:## or ##:##:##')
        else:
            start = int(values[0])
            if len(values) == 3:
                step = int(values[1])
                end = int(values[2])
            elif len(values) == 2:
                step = 1
                end = int(values[1])
            else:
                raise RuntimeError('If : is used, vals must be ##:## or ##:##:##')
        vals_nums = np.arange(start, end, step)
        out = np.array([vali for vali in vals_nums], dtype=dtype)
    else:
        out = np.array([vals.split('/')[i] for i in range(len(vals.split('/')))], dtype=dtype)

    return out


def float2pstr(floatv, ndigits=2):
    """Format a float as a string, replacing decimal points with p"""
    return ('{0:0.' + str(int(ndigits)) + 'f}').format(floatv).replace('.', 'p').replace('-', 'n')


def str2float(string):
    """Format a string as a float, ensuring to replace p and n with . and -"""
    return float(string.replace('p', '.').replace('n', '-'))


def array2string_sequence(arr, pn_replacement=False, ndigits=3):
    """Format a float array as a string, optionally replace p and n with . and -"""
    if pn_replacement:
        outstr = ''
        kk = 0
        for floatv in arr:
            outstr += float2pstr(floatv, ndigits=3)
            if kk < len(arr) - 1:
                outstr += '/'
            kk += 1
    else:
        outstr = ''
        kk = 0
        for floatv in arr:
            outstr += ('{0:0.' + str(int(ndigits)) + 'f}').format(floatv)
            if kk < len(arr) - 1:
                outstr += '/'
            kk += 1
    return outstr


def prepstr(string):
    """Format a string for using as part of a directory string"""
    return string.replace('.', 'p').replace('-', 'n')
