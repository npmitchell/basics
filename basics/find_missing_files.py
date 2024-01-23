import glob
import os
import argparse
import basics.dataio as dio
import numpy as np

"""
Look through a directory for a sequence of files with increasing indices in the filename. Record those that are missing.

Examples
--------
python /mnt/data/code/basics/basics/find_missing_files.py -fn \
    Time_wildcard_Angle_0_c1_ls_1.ome.tif -start 0 -end 209 -padding 6

python /mnt/data/code/basics/basics/find_missing_files.py -fn \
    deconvolved_32bit_4view_thres0p140_pix2_sig15_affinereg_individual_registration/TPwildcard_Ch0_Ill0_Ang0,45,90,135,180,225,270,315.tif -start 0 -end 209 

python /mnt/data/code/basics/basics/find_missing_files.py -fn \
    Time_wildcard_c1.tif -start 0 -end 209 -padding 6
    
"""

parser = argparse.ArgumentParser(description='Create movies color coded by depth')
parser.add_argument('-d', '--datadir', help='Directory with data', type=str, default='./')
parser.add_argument('-fn', '--filename',
                    help="file name in datadir, with 'wildcard' in place of indices to change",
                    type=str, default='empty_string')
parser.add_argument('-start', '--start', help='First integer index for seeking filename',
                    type=int, default=0)
parser.add_argument('-end', '--end', help='Last integer index for seeking filename',
                    type=int, default=100)
parser.add_argument('-padding', '--padding', help='If >0, how many digits the index is', type=int, default=0)
args = parser.parse_args()

searchstr = os.path.join(args.datadir, args.filename.replace('wildcard', '*'))
print('Seeking fns: ' + searchstr)
fns = sorted(glob.glob(searchstr))
if len(fns) > 0:
    print('Found fns', fns[0] + ', ...')
else:
    raise RuntimeError('Found no files!')

# For splitting the fn
fn_part0 = args.filename.split('wildcard')[0]
fn_part1 = args.filename.split('wildcard')[1]


# get old index from fn
found_some = False
for index in np.arange(args.start, args.end):
    fn = args.datadir + fn_part0 + str(index).zfill(args.padding) + fn_part1

    if not glob.glob(fn):
        print('Did not find: ' + str(index).zfill(args.padding))
        found_some = True

if not found_some:
    print('Did not find any missing files!')
