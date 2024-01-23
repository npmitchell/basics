# !/usr/bin/python
import os, sys
import glob
import argparse

# from os import getcwd,chdir

'''Replace unique character substring in all filenames or directory names in a specified directory.

Example usage (all valid):
python rename_files.py -oldstr textfile -newstr nicefile -dir ./directory_to_change
python rename_files.py -oldstr textfile -newstr nicefile -dir ./directory_to_change/
python rename_files.py -oldstr textfile -newstr nicefile -dir /Users/labuser/directory_to_change
python rename_files.py -oldstr textfile -newstr nicefile -dir /Users/labuser/directory_to_change/
python rename_files.py -oldstr textfile -newstr nicefile -dir ./directory_to_change/,./other_dir_to_change/
python rename_files.py -oldstr msls_apical_0 -newstr msls_apical_stab_0 -dir /mnt/crunch/48Ygal4-UAShistRFP/201904031830_great/Time4views_60sec_1p4um_25x_1p0mW_exp0p35_2/data/deconvolved_16bit/msls_output_prnun5_prs1_nu0p00_s0p10_pn2_ps4_l1_l1/

Will NOT work:
python rename_files.py -oldstr textfile -newstr nicefile -dir ../directory_to_change/
python rename_files.py -oldstr textfile -newstr nicefile -dir ../../directory_to_change/

'''


def rename_files(directory, oldstr, newstr):
    """Replace all the filenames in directory containing oldstr with newstr.
    
    Parameters
    ----------
    directory : string
        The directory in which to look for files with filenames containing oldstr
    oldstr : string
        string to replace in filenames
    newstr : string
        string to put in place of oldstr in filenames
    """
    # Get directory name from input arguments
    pwd = os.getcwd() + '/'
    directory = directory.replace('./', pwd)
    if directory[-1] != '/':
        directory += '/'

    print("The directory is: %s" % directory)
    print("The files to be replaced are is: %s" % glob.glob(directory + '*' + oldstr + '*'))

    # Note below we avoid overwritting THIS file
    # (which could be changed if looking in our present working directory).
    for filename in glob.glob(directory + '*' + args.oldstr + '*'):
        if not filename == 'rename_files.py':
            print('filename = ', filename)
            # renaming files in directory
            if args.oldstr in filename:
                newname = filename.split(oldstr)[0] + newstr + filename.split(oldstr)[1]
                os.rename(filename, newname)
                print('new filename = ', newname)

    print("Successfully renamed.")

    # listing directories after renaming "tutorialsdir"
    print("the dir now contains: %s" % os.listdir(directory))


if __name__ == '__main__':
    # Parse arguments from command line
    parser = argparse.ArgumentParser(description='Specify time string (timestr) for gyro simulation.')
    parser.add_argument('-dir', '--directory', help='Name of directory inside which to rename files.' +
                                                    'If not absolute path, use ./ to denote pwd. ' +
                                                    'Currently cannot contain ../', type=str, default='./')
    parser.add_argument('-oldstr', '--oldstr', help='String to delete in file names', type=str, default='')
    parser.add_argument('-newstr', '--newstr', help='String to add in file names in place of delete',
                        type=str, default='')
    args = parser.parse_args()

    # For each directory specified by dir, run rename_files
    dirs = args.directory.split(',')

    for directory in dirs:
        rename_files(directory, args.oldstr, args.newstr)
