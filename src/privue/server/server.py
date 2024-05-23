"""LDP - SUE Server Side

This submodule contains all the relevant functions for estimation of privatized data on the server side (privatization using client side submodule).
Use the functions in this module to estimate distribution and average across client values with the SUE method.
The method requires the following FIXED parameters, which must be consistent between the server and clients:
    Epsilon - the privacy budget
    For each value, the next parameters must be fixed separately:
        Data Range - defined by [Min. value, Max value]
        Bucket Amount (d) - desired number of even sub-sections of the the Data Range (We refer to a subsection as "bucket")
        Example: Data range = [0, 100], Bucket Amount = 5 => Sub-sections result: [0,20),[20-40),[40-60),[60-80),[80-100]
Estimate a value distribution and average once you have collected the privatized d (bucket amount) length histogram from N clients across L timestamps in your database of choice.
The function for estimation require you to extract the collected data into a numpy NDarray the shape of NxLxd for each separate value your user submits.

Example: Estimation of sugar percentage & calorie amount of clients' meals during a week in a privatized form (Following the example in client side documentation):
    You have collected privatized data of sugar percentage & calorie amount from 500 clients' meals during a whole week - 7 days (each day is considered a timestamp)
    Chosen epsilon - privacy budget = 2
    Sugar percentage value properties: 
        Data range = [0, 100]
        Bucket Amount = 5
    Calorie amount value properties: 
        Data range = [0, 1000]
        Bucket Amount = 10
    Extract the collected data into a numpy NDarray for each value: 
        For sugar percentage: the shape of NDarray must be 500x7x5
        For calorie amount: the shape of NDarray must be 500x7x10
        * If a user hasn't submitted a value during a particular timestamp, then the relevant d length numpy array in the tensor entry must be a np.full(d, np.nan)
    Now run the estimation function with the list of these 2 tensors and the correspondent properties.
    

This script requires that `numpy` be installed within the Python
environment you are running this script in.
"""

import math
import numpy as np
import privue.client.client as client
from privue.errors.errors import ValuePropertyMissing
from typing import List


def get_avg_vector_estimation(tensor : np.ndarray, epsilon: float):
    """Calculates an estimation of the average histogram.

    Args:
        tensor: NxLxd NDarray - output of the privatization process as described in the preamble
        epsilon: Privacy budget

    Returns:
        The estimation of the average histogram - represents the distribution of the original data, using only the privatized data.
    """
    n, l, d = tensor.shape
    temp = math.exp(epsilon/2)
    p = temp / (temp + 1)
    temp = np.sum(tensor, axis=2)
    nan_count = np.isnan(temp).sum()
    v_hat_sum_timestamp_axis = np.nansum(tensor, axis=1)
    v_hat_sum = np.nansum(v_hat_sum_timestamp_axis, axis=0)
    return ( ( v_hat_sum / (n * l - nan_count) ) - (1-p) ) / (2*p-1)


def get_weights_vector(max_value : float, min_value : float, bucket_amount : int):
    """Calculates a weights vector - each cell contains the mean value of the corresponding bucket.

    Args:
        max_value: Max value in the Data Range
        min_value: Min. value in the Data Range
        bucket_amount: Amount of sub-sections (buckets) in Data Range

    Returns:
        The weights vector the size of bucket_amount - each cell contains the mean value of the corresponding bucket.

        Example: Data range = [0, 100], Bucket Amount = 5 => Weights Vector = (10,30,50,70,90) 
    """
    granularity = client.get_bucket_granularity(max_value, min_value, bucket_amount)
    return np.array([(min_value + granularity * (i + 0.5)) for i in range(bucket_amount)])


def average_estimation(tensor : np.ndarray, epsilon : float, max_value : float, min_value : float, bucket_amount : float, return_avg_histogram : bool = False) -> float | tuple[float, np.ndarray]:
    """Estimates the average of the original data.

    Args:
        tensor: NxLxd NDarray - output of the privatization process as described in the preamble
        epsilon: Privacy budget
        max_value: Max value in the Data Range
        min_value: Min. value in the Data Range
        return_avg_histogram (optional): Specifying whether to return the average histogram estimation alongside the average estimation
    
    Returns:
        if return_avg_histogram = True - return a list:
            the first element is the estimated average of the original data, using only the privatized data.
            the second element is the average histogram - the result of get_avg_vector_estimation function
        else:
            return a single float - the estimated average
    """
    weights_vector = get_weights_vector(max_value, min_value, bucket_amount)
    avg_vector = get_avg_vector_estimation(tensor, epsilon)
    if return_avg_histogram:
        return (np.nansum(np.multiply(weights_vector,avg_vector)), avg_vector)
    return np.nansum(np.multiply(weights_vector,avg_vector))


def average_estimation_multiple_attr(tensor_list : List[np.ndarray], epsilon : List[float], max_value_per_attr_list : List[float], min_value_per_attr_list : List[float], bucket_amount_per_attr_list : List[int], return_avg_histogram : bool = False) -> List[float] | List[tuple[float, np.ndarray]]:
    """Estimates the average of the original data, for each client value.

    This function is a generalized version of the average_estimation function. 
    Instead of receiving NDarray and it's properties (Data Range, Bucket amount), it receives a list for each property and a list of NDarray which correspond in their order.
    The epsilon privacy budget is shared across all values, as described in the spl technique - guaranteeing correct estimation.

    Args:
        tensor_list: list of NxLxd NDarrays, with corresponding d for each value - outputs of the privatization process as described in the preamble
        epsilon: Privacy budget
        max_value_per_attr_list: A list with the Max value of a Data Range for each value
        min_value_per_attr_list: A list with the Min. value of a Data Range for each value
        bucket_amount_per_attr_list: A list with the desired number of sub ranges (buckets) for each value
        return_avg_histogram (optional): Specifying whether to return the average histogram estimation alongside the average estimation

    Returns:
        if return_avg_histogram = True - return a list of average_estimation results for each value:
            the first element in a result is the estimated average of the original data, using only the privatized data, for the particular value
            the second element in a result is the average histogram - the result of get_avg_vector_estimation function, for the particular value
        else:
            return a list of floats - the estimated average for each particular value
            
    Raises:
        ValuePropertyMissing: If a value is missing one of it's properties (number of values must be non-zero)
    """
    k = len(max_value_per_attr_list)
    if len(min_value_per_attr_list) != k or len(bucket_amount_per_attr_list) != k:
        raise ValuePropertyMissing(f"One of your values is missing a property - Max value property list length {k}, Min. value property list length {len(min_value_per_attr_list)}, Bucket number property list length {len(bucket_amount_per_attr_list)}")
    if k == 0:
        raise ValuePropertyMissing("Must have at least 1 value with corresponding properties: max value, min. value and bucket number")
    res = []
    for i in range(k):
        res.append(average_estimation(tensor_list[i], (epsilon/k), max_value_per_attr_list[i], min_value_per_attr_list[i], bucket_amount_per_attr_list[i], return_avg_histogram))
    return res