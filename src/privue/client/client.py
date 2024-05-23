"""LDP - SUE Client Side

This submodule contains all the relevant functions for privatization on the client side.
Each client submits a single/multiple values to be collected by the server (in a single timestamp).
Use the functions in this module to privatize the values with the SUE method.
The method requires the following FIXED parameters, which must be consistent between the server and clients:
    Epsilon - the privacy budget
    For each value, the next parameters must be fixed separately:
        Data Range - defined by [Min. value, Max value]
        Bucket Amount - desired number of even sub-sections of the the Data Range (We refer to a subsection as "bucket")
        Example: Data range = [0, 100], Bucket Amount = 5 => Sub-sections result: [0,20),[20-40),[40-60),[60-80),[80-100]

Example: Collection of sugar percentage & calorie amount of users' meals during a week in a privatized form:
    Chosen epsilon - privacy budget = 2
    Sugar percentage value properties: 
        Data range = [0, 100]
        Bucket Amount = 5
    Calorie amount value properties: 
        Data range = [0, 1000]
        Bucket Amount = 10
    The client submits his actual values of 38 sugar grams and 800 calories respectively.
    The random privatization function returns a list of privatized histograms (vectors) representing each of original values:
        For sugar grams: [1,0,1,1,0]
        For calories: [0,0,0,1,1,1,0,0,1,0]
        * Notice how the size of each privatized vector corresponds to the bucket amount of the particular value
    Now send to the server side the privatized histograms instead and save it there in your database of choice.

This script requires that `numpy` be installed within the Python
environment you are running this script in.
"""

import math
import numpy as np
from privue.errors.errors import ValuePropertyMissing
from typing import List, Dict


def get_bucket_granularity(max_value : float, min_value : float, bucket_amount : int) -> float: 
    """Calculates the granularity of a bucket in the Data range.

    Args:
        max_value: Max value in the Data Range
        min_value: Min. value in the Data Range
        bucket_amount: Amount of sub-sections (buckets) in Data Range

    Returns:
        The granularity of a bucket

        Example: Data range = [0, 100], Bucket Amount = 5 => Granularity = 20
    """
    return (max_value - min_value) / bucket_amount


def get_bucket_index_by_value(max_value : float, min_value : float, bucket_amount : int, client_value : float) -> int:
    """Calculates the index of the sub-section (bucket) where the client value belongs.

    Args:
        max_value: Max value in the Data Range
        min_value: Min. value in the Data Range
        bucket_amount: Amount of sub-sections (buckets) in Data Range
        client_value: Client's value

    Returns:
        Index of the sub-section (bucket) where the client value belongs.

        Example: Data range = [0, 100], Bucket Amount = 5, Client Value = 66 => Index = 3 ( Sub section [60-80) )
    """
    granularity = get_bucket_granularity(max_value, min_value, bucket_amount)
    if client_value < min_value:
        return 0
    elif client_value > max_value:
        return bucket_amount - 1
    else:
        return math.floor((client_value - min_value) / granularity)
    

def get_true_vector(max_value : float, min_value : float, bucket_amount : int, client_value : float) -> np.ndarray: #!
    """Calculates a binary histogram (vector), representing the given value.

    Args:
        max_value: Max value in the Data Range
        min_value: Min. value in the Data Range
        bucket_amount: Amount of sub-sections (buckets) in Data Range
        client_value: Client's value

    Returns:
        A binary vector with value 1 in the index of the sub-section (bucket) where the value belongs, 0 in other indexes.

        Example: Data range = [0, 100], Bucket Amount = 5, Client Value = 66 => [0, 0, 0, 1, 0] (1 in the relevant bucket)
    """
    vector = np.zeros(bucket_amount)
    vector[get_bucket_index_by_value(max_value, min_value, bucket_amount, client_value)] = 1
    return vector
    
    
def get_private_vector(epsilon : float, max_value : float, min_value : float, bucket_amount : int, value : float, memo_dict : Dict[int, np.ndarray] = None) -> np.ndarray:
    """Calculates a privatized binary histogram (vector) using the SUE-LDP algorithm.

    The function supports a memoization dictionary to maintain privacy. 
    The dictionary must be kept on the client side in a dedicated data structure and imported every time the function is used.
    The function updates the dictionary, therefore update it in the dedicated data structure, otherwise the privacy will be corrupted.

    Args:
        epsilon: Privacy budget
        max_value: Max value in the Data Range
        min_value: Min. value in the Data Range
        bucket_amount: Amount of sub-sections (buckets) in Data Range
        client_value: Client's value
        memo_dict (optional): Imported dictionary for memoization

    Returns:
        A privatized binary vector, which satisfies epsilon-LDP definition.
    """
    if np.isnan(value):
        return np.nan
          
    bucket_index = get_bucket_index_by_value(max_value, min_value, bucket_amount, value)
    
    if memo_dict is not None and bucket_index in memo_dict:
        return memo_dict[bucket_index]
    
    vector = np.zeros(bucket_amount)
    temp = math.exp(epsilon/2)
    
    for i in range(bucket_amount):
        if i == bucket_index:
            p_numerator  = temp
        else:
            p_numerator = 1
        p = p_numerator / (temp + 1)
        vector[i] = np.random.binomial(1,p,1)
        
    if memo_dict is not None:
        memo_dict[bucket_index] = vector
        
    return vector


def get_private_vector_multiple_attr(epsilon : float, max_value_per_attr_list: List[float], min_value_per_attr_list: List[float], bucket_amount_per_attr_list: List[int], client_value_per_attr_list: List[float], memo_dict_per_attr: List[Dict[int, np.ndarray]] = None) -> List[np.ndarray]:
    """Calculates a list of privatized binary histograms (vectors) using the SUE-LDP algorithm, and the Spl technique, for each client value.

    This function is a generalized version of the get_private_vector function. 
    Instead of receiving a single value and it's properties (Data Range, Bucket amount), it receives a list for each property and a list of values which correspond in their order.
    The epsilon privacy budget is shared across all values, as described in the spl technique - guaranteeing the entire array of privatized values is epsilon-LDP.
    The function supports memoization dictionaries to maintain privacy, just like get_private_vector. 
    Each dictionary must be kept on the client side in a dedicated data structure and imported every time the function is used.
    The function updates all dictionaries, therefore update them in the dedicated data structure, otherwise the privacy will be corrupted.

    Args:
        epsilon: Privacy budget
        max_value_per_attr_list: A list with a Max value of a Data Range for each value
        min_value_per_attr_list: A list with a Min. value of a Data Range for each value
        bucket_amount_per_attr_list: A list with a desired number of sub ranges (buckets) for each value
        client_value_per_attr_list: A list of the client's values
        memo_dict_per_attr (optional): A list of imported memoization dictionaries, separate for each value

    Returns:
        An array of privatized binary vectors, which satisfies epsilon-LDP definition.
        
    Raises:
        ValuePropertyMissing: If a value is missing one of it's properties (number of values must be non-zero)
    """
    k = len(max_value_per_attr_list)
    if len(min_value_per_attr_list) != k or len(bucket_amount_per_attr_list) != k or len(client_value_per_attr_list) != k:
        raise ValuePropertyMissing(f"One of your values is missing a property - Max value property list length {k}, Min. value property list length {len(min_value_per_attr_list)}, Bucket number property list length {len(bucket_amount_per_attr_list)}, User value property list length {len(client_value_per_attr_list)}")
    if k == 0:
        raise ValuePropertyMissing("Must have at least 1 value with corresponding properties: max value, min. value, bucket number, user value")
    if memo_dict_per_attr is None:
        memo_dict_per_attr = [None for _ in range(k)]
    new_epsilon =  epsilon / k
    return [get_private_vector(new_epsilon, max_value_per_attr_list[i], min_value_per_attr_list[i], bucket_amount_per_attr_list[i], client_value_per_attr_list[i], memo_dict_per_attr[i]) for i in range(k)]