import os
def chunks(l, n):
    """ Yield successive n-sized chunks from l.
        from http://goo.gl/DZNhk
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def make_dir_if_not_present(directory):
    """
        Create directory if it does not exist yet.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
