def get_time_in_hms(dt):
    m, s = divmod(dt, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), s

def arange_generator(m):
    '''this method provides a generator as a alternative to the np.arange method'''
    assert type(m) == int, ''
    assert m >= 0, ''

    n = 0
    while n < m:
        yield n
        n += 1

