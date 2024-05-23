"""LDP - SUE Utility

This submodule contains all the general utility functions which are used for the demo and JSON file format privatization support.

About JSON format:
Required format for privatization: JSON string must be comprised of main JSON Objects - each represents a unique user. 
Every user's object has nested JSON objects which represent timestamps. 
The numeric values for each timestamp are held in an array (consistent order for every timestamp). 
If a user doesn't have a record for a timestamp, it shouldn't appear in the user object. Here is an example string (3 values per user record):
See the "privatization_json_example" file in the Github directory.

Required format for estimation: JSON string must be comprised of 3 main JSON Objects:
    "epsilon", which is the value of the privacy budget.
    "attr_data", which contains an array of sub-arrays (length 3) - [max value, min. value, bucket amount] of the particular value
    "data", an object similar to the format for privatization. Instead of each record being the real value, it is the privatized histogram of it (result of client.get_private_vector) 
See the "estimation_json_example" file in the Github directory, which is a privatization of the "privatization_json_example" file.
"""


from io import StringIO
import json
import numpy as np
import pandas as pd
import privue.client.client as client
import privue.server.server as server
from typing import List, Dict


def privatize_json_string(input_json_str : str, epsilon : float, max_value_per_attr_list : List[float], min_value_per_attr_list : List[float], bucket_amount_per_attr_list : List[int]):
    """Returns a JSON string of the required format for estimation

    This function is a generalized version of the get_private_vector_multiple_attr function,
    which accommodates JSON string, in the specified format for privatization, as input. 
    It receives a list for each property and a list of values which correspond in their order.
    The epsilon privacy budget is shared across all values, as described in the spl technique - guaranteeing the entire array of privatized values is epsilon-LDP.

    Args:
        input_json_str: a JSON string of the required format for privatization
        epsilon: Privacy budget
        max_value_per_attr_list: A list with a Max value of a Data Range for each value
        min_value_per_attr_list: A list with a Min. value of a Data Range for each value
        bucket_amount_per_attr_list: A list with a desired number of sub ranges (buckets) for each value

    Returns:
        An array of privatized binary vectors, which satisfies epsilon-LDP definition.
        
    Raises:
        ValuePropertyMissing: If a value is missing one of it's properties (number of values must be non-zero)
    """
    def privatize(worker_values):
        if type(worker_values) != list and pd.isna(worker_values):
            return None
        return client.get_private_vector_multiple_attr(epsilon, max_value_per_attr_list, min_value_per_attr_list, bucket_amount_per_attr_list,worker_values)
    df = pd.read_json(StringIO(input_json_str))
    df = df.applymap(privatize)
    json_str = df.to_json()
    parsed_json = json.loads(json_str)
    privatized_json = {"data": parsed_json}
    privatized_json["epsilon"] =  epsilon
    privatized_json["attr_data"] =  [(max_value_per_attr_list[i], min_value_per_attr_list[i], bucket_amount_per_attr_list[i]) for i in range(len(max_value_per_attr_list))]
    return json.dumps(privatized_json)
        
        
def privatize_json_file(input_json_file_path : str, epsilon : float, max_value_per_attr_list : List[float], min_value_per_attr_list : List[float], bucket_amount_per_attr_list : List[int]):
    """Returns a JSON string of the required format for estimation (using filepath)

    This function is a generalized version of the get_private_vector_multiple_attr function,
    which accommodates JSON string, in the specified format for privatization, as input. 
    It receives a list for each property and a list of values which correspond in their order.
    The epsilon privacy budget is shared across all values, as described in the spl technique - guaranteeing the entire array of privatized values is epsilon-LDP.

    Args:
        input_json_file_path: path for JSON file of the required format for privatization
        epsilon: Privacy budget
        max_value_per_attr_list: A list with a Max value of a Data Range for each value
        min_value_per_attr_list: A list with a Min. value of a Data Range for each value
        bucket_amount_per_attr_list: A list with a desired number of sub ranges (buckets) for each value

    Returns:
        An array of privatized binary vectors, which satisfies epsilon-LDP definition.
        
    Raises:
        ValuePropertyMissing: If a value is missing one of it's properties (number of values must be non-zero)
    """
    with open(input_json_file_path, 'r') as input_json_str:
        input_json_str = input_json_str.read()
    return privatize_json_string(input_json_str, epsilon, max_value_per_attr_list, min_value_per_attr_list, bucket_amount_per_attr_list)
   
        
def _get_tensor_list_from_privatized_json_obj(input_json_obj : Dict) -> List[np.ndarray]:
    """Returns a NDarray list required for average estimation from JSON object.

    Args:
        input_json_obj: a JSON object of the required format for estimation

    Returns:
        NDarray list (each shaped NxLxd, each d corresponding to specific number of buckets) required for average estimation
    """
    data_dict = input_json_obj["data"]
    attr_bucket_number_list = [attr_data[2] for attr_data in input_json_obj["attr_data"]]

    user_dict_keys = data_dict.keys()

    if len(user_dict_keys) == 0:
        raise AttributeError("No JSON object if file.")

    first_key = list(user_dict_keys)[0]
    timestamp_dict_keys = data_dict[first_key].keys()
    attr_num = len(attr_bucket_number_list)
    n_l_k_tensor_list_per_attr = []
    for attr_data_index in range(attr_num):
        specific_attr_matrix_list_per_user = []
        for user_key in user_dict_keys:
            specific_attr_histogram_array_list_per_timestamp = []
            for timestamp_key in timestamp_dict_keys:
                curr_list = data_dict[user_key][timestamp_key]
                if curr_list is None:
                    curr_array = np.full(attr_bucket_number_list[attr_data_index], np.nan)
                else:
                    curr_array = np.array(curr_list[attr_data_index])
                specific_attr_histogram_array_list_per_timestamp.append(curr_array)    
            specific_attr_matrix_list_per_user.append(np.array(specific_attr_histogram_array_list_per_timestamp)) # append specific attribute - specific worker - Lxk matrix
        n_l_k_tensor_list_per_attr.append(np.array(specific_attr_matrix_list_per_user)) # append specific attribute - NxLxk tensor
    return n_l_k_tensor_list_per_attr


def get_tensor_list_from_privatized_json_str(input_json_str : str) -> List[np.ndarray]:
    """Returns a NDarray list required for average estimation from JSON string.

    Args:
        input_json_str: a JSON string of the required format for estimation

    Returns:
        NDarray list (each shaped NxLxd, each d corresponding to specific number of buckets) required for average estimation
    """
    input_json_obj = json.loads(input_json_str)
    return _get_tensor_list_from_privatized_json_obj(input_json_obj)


def avg_estimation_with_json_str(input_json_str : str, return_avg_histogram : bool = False) -> List[np.ndarray]:
    """Estimates the average of the original data, for each client value.

    This function is a version of the server.average_estimation_multiple_attr function,
    which accommodates JSON string, in the specified format for estimation, as input.

    Args:
        input_json_str: a JSON string of the required format for estimation
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
    
    input_json_obj = json.loads(input_json_str)
    attr_data_list = input_json_obj["attr_data"]
    epsilon = input_json_obj["epsilon"]
    tensor_list = _get_tensor_list_from_privatized_json_obj(input_json_obj)
    max_value_per_attr_iter = [attr_data[0] for attr_data in attr_data_list]
    min_value_per_attr_iter = [attr_data[1] for attr_data in attr_data_list]
    bucket_amount_per_attr_iter = [attr_data[2] for attr_data in attr_data_list]
    return server.average_estimation_multiple_attr(tensor_list, epsilon, max_value_per_attr_iter, min_value_per_attr_iter, bucket_amount_per_attr_iter, return_avg_histogram)


def avg_estimation_with_json_file(input_json_file, return_avg_histogram=False):
    """Estimates the average of the original data, for each client value (using filepath).

    This function is a version of the server.average_estimation_multiple_attr function,
    which accommodates JSON string, in the specified format for estimation, as input.

    Args:
        input_json_file: path for JSON file of the required format for estimation
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
    with open(input_json_file, 'r') as json_file:
        input_json_str = json_file.read()
    return avg_estimation_with_json_str(input_json_str, return_avg_histogram)
    
    
def get_granularity_dataframe(max_value : float, min_value : float, bucket_amount : int):
    """Outputs pandas dataframe detailing the range of each sub section

    Args:
        max_value: Max value in the Data Range
        min_value: Min. value in the Data Range
        bucket_amount: Amount of sub-sections (buckets) in Data Range

    Returns:
        pandas dataframe detailing the range of each sub section
    """
    granularity = client.get_bucket_granularity(max_value, min_value, bucket_amount)
    granularity_dict = {str(j): [f"{j * granularity}-{(j+1) * granularity}"] for j in range(bucket_amount)}
    df = pd.DataFrame.from_dict(granularity_dict)  
    return df