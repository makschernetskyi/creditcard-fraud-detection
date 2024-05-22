def add_prefix_to_dict_keys(d, prefix):
    """
    Add a prefix to each key in the dictionary.

    Parameters
    ----------
    d : dict
        The input dictionary whose keys need to be prefixed.
    prefix : str
        The prefix to add to each key in the dictionary.

    Returns
    -------
    dict
        A new dictionary with the prefixed keys.
    """
    return {prefix + str(key): value for key, value in d.items()}

