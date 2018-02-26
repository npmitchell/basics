# !/usr/bin/python
import os, sys
import glob
import argparse
import lepm.dataio as dio
# from os import getcwd,chdir

'''Remove file for all filenames or directory names in a specified directory.
USE CAUTIOUSLY!

# Often used for:
python remove_files.py -strspec eigvect.pkl -dir /Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/iscentroid/ -subdirspec iscentroid_square*
python remove_files.py -strspec eigvect_mass.pkl -dir /Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/iscentroid/ -subdirspec iscentroid_square*

Example usage (all valid):
python remove_files.py -strspec badfile -dir ./directory_to_change
python remove_files.py -strspec badfile -dir ./directory_to_change/
python remove_files.py -strspec badfile -dir /Users/labuser/directory_to_change
python remove_files.py -strspec badfile -dir /Users/labuser/directory_to_change/
python remove_files.py -strspec badfile -dir ./directory_to_change/,./other_dir_to_change/

Will NOT work:
python remove_files.py -strspec badfile -dir ../directory_to_change/
python remove_files.py -strspec badfile -dir ../../directory_to_change/
'''


def remove_files(directory, strspec):
    """Delete all the filenames in directory containing strspec.
    
    Parameters
    ----------
    directory : string
        The directory in which to look for files with filenames containing oldstr
    strspec : string
        string to find in filenames to delete
    """
    # Get directory name from input arguments
    pwd = os.getcwd()+'/'
    directory = directory.replace('./',pwd)
    if directory[-1] != '/':
        directory += '/'
    
    print "The directory is: %s" % directory
    print "The files to be removed are: %s" % glob.glob(directory + strspec)
    
    # Note below we avoid deleting THIS file
    # (which could be changed if looking in our present working directory).
    for filename in glob.glob(directory + strspec):
        if not filename == 'rename_files.py':
            # remove files in directory
            if strspec in filename:
                print 'deleting filename = ', filename
                raw_input("Press Enter to continue...")
                os.remove(filename)
    
        print "Successfully removed."


if __name__ == '__main__':
    # Parse arguments from command line
    parser = argparse.ArgumentParser(description='Specify time string (timestr) for gyro simulation.')
    parser.add_argument('-dir', '--directory', help='Name of directory inside which to rename files.' +
                                                    'If not absolute path, use ./ to denote pwd. ' +
                                                    'Currently cannot contain ../', type=str, default='./')
    parser.add_argument('-strspec', '--strspec', help='String to search for in file names', type=str, default='')
    parser.add_argument('-subdirspec', '--subdirspec', help='String to find in subdirectory names',
                        type=str, default='check_string_for_empty')
    args = parser.parse_args()
    
    # For each directory specified by dir, run rename_files
    dirs = args.directory.split(',')
    
    for directory in dirs:
        if args.subdirspec != 'check_string_for_empty':
            sdirs = dio.find_subdirs(args.subdirspec, directory)
            for dir in sdirs:
                remove_files(dir, args.strspec)
        else:
            remove_files(directory, args.strspec)


    
    

