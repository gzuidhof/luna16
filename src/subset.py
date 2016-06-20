import glob
from sklearn.externals import joblib

def make_filename_to_subset_dict():

    in_folder = "/scratch-shared/vdgugten/cad/original/"
    subset_dict = {}

    for subset in range(10):
        files = glob.glob(in_folder+'subset{0}'.format(subset)+'/*.mhd')
        print "Subset", subset, "n files", len(files)
        files = map(lambda x: x.replace('.mhd','').split('/')[-1], files)
        subset_dict[subset] = files

    joblib.dump(subset_dict, '../config/subset_to_filenames.pkl')

def get_subset_to_filename_dict(location='../../config/subset_to_filenames.pkl'):
    return joblib.load(location)

if __name__ == "__main__":
    make_filename_to_subset_dict()
