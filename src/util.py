def chunks(l, n):
    """ Yield successive n-sized chunks from l.
        from http://goo.gl/DZNhk
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
