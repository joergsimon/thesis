
import itertools


# TODO: for the general case this is fine, but f.e. for correlation this counts some things double.
def get_tuples(values):
    tupels = [(a, b) for a in values for b in values if a != b]
    return tupels


def get_combinations(values):
    combinations = itertools.combinations(values, 2)
    return combinations