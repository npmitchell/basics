import glob
import os
import argparse
import basics.dataio as dio

"""
Rename a sequence of files, shifting their filename index by a given value.
To overwrite the original files, let the datadir == outdir (ie do not specify the outdir)
If a file will be overwritten by this procedure, for ex if increasing the index by some value (0->2, 1->3, etc),
then an error will be thrown to avoid losing data.
Right now, this just works for integer values, but could generalize to floats.

Examples
--------
python /mnt/data/code/basics/basics/batch_reindex_files.py -fn mip_1_wildcard_c1.tif -shift 5 -o ./shifted/

"""

parser = argparse.ArgumentParser(description='Create movies color coded by depth')
parser.add_argument('-d', '--datadir', help='Directory with data', type=str, default='./')
parser.add_argument('-o', '--outdir', help='Output directory, if not datadir', type=str, default='empty_string')
parser.add_argument('-fn', '--filename',
                    help="file name in datadir, with 'wildcard' in place of indices to change",
                    type=str, default='empty_string')
parser.add_argument('-shift', '--shift',
                    help='Index shift value -- how much to adjust the indices in filename. '
                         'To shift by a negative amount, lead the integer with m for minus.',
                    type=str, default="0")
args = parser.parse_args()

searchstr = os.path.join(args.datadir, args.filename.replace('wildcard', '*'))
print('Seeking fns: ' + searchstr)
fns = sorted(glob.glob(searchstr))
if len(fns) > 0:
    print('Found fns', fns[0] + ', ...')
else:
    raise RuntimeError('Found no files!')

# Prepare the output directory
outdir = args.outdir
dio.ensure_dir(outdir)

# For splitting the fn
fn_part0 = args.filename.split('wildcard')[0]
fn_part1 = args.filename.split('wildcard')[1]

# Get the index shift
shift = int(args.shift.replace('m', '-'))

for fn in fns:
    # get old index from fn
    indexstr = fn.split(fn_part0)[-1].split(fn_part1)[0]
    index = int(indexstr)
    dd = len(indexstr)
    newfn = outdir + fn_part0 + str(index + shift).zfill(dd) + fn_part1

    if fn != newfn:
        if os.path.exists(newfn):
            raise RuntimeError('File already exists: ' + newfn)
        else:
            print(fn + '\n -> ' + newfn)
            command = 'cp ' + fn + ' ' + newfn
            os.system(command)

