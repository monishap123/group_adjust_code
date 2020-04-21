import pytest
from datetime import datetime
import numpy as np
import pandas as pd

# Your task is to write the group adjustment method below. There are some
# unimplemented unit_tests at the bottom which also need implementation.
# Your solution can be pure python, pure NumPy, pure Pandas
# or any combination of the three.  There are multiple ways of solving this
# problem, be creative, use comments to explain your code.

# Group Adjust Method
# The algorithm needs to do the following:
# 1.) For each group-list provided, calculate the means of the values for each
# unique group.
#
#   For example:
#   vals       = [  1  ,   2  ,   3  ]
#   ctry_grp   = ['USA', 'USA', 'USA']
#   state_grp  = ['MA' , 'MA' ,  'CT' ]
#
#   There is only 1 country in the ctry_grp list.  So to get the means:
#     USA_mean == mean(vals) == 2
#     ctry_means = [2, 2, 2]
#   There are 2 states, so to get the means for each state:
#     MA_mean == mean(vals[0], vals[1]) == 1.5
#     CT_mean == mean(vals[2]) == 3
#     state_means = [1.5, 1.5, 3]
#
# 2.) Using the weights, calculate a weighted average of those group means
#   Continuing from our example:
#   weights = [.35, .65]
#   35% weighted on country, 65% weighted on state
#   ctry_means  = [2  , 2  , 2]
#   state_means = [1.5, 1.5, 3]
#   weighted_means = [2*.35 + .65*1.5, 2*.35 + .65*1.5, 2*.35 + .65*3]
#
# 3.) Subtract the weighted average group means from each original value
#   Continuing from our example:
#   val[0] = 1
#   ctry[0] = 'USA' --> 'USA' mean == 2, ctry weight = .35
#   state[0] = 'MA' --> 'MA'  mean == 1.5, state weight = .65
#   weighted_mean = 2*.35 + .65*1.5 = 1.675
#   demeaned = 1 - 1.675 = -0.675
#   Do this for all values in the original list.
#
# 4.) Return the demeaned values

# Hint: See the test cases below for how the calculation should work.

"""
Internal function to check if a string is a valid number 

Parameters
----------
str: a string indicating a number

Returns
-------
boolean value if indicating if input str is a valid floating point value or not
"""
def __is_number(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

"""
Internal function to compute weighted mean 

Parameters
----------
row     : A dict of with keys ['group_0', 'value_0', 'group_1', 
'value_1', 'group_2', 'value_2'] and their corresponding values
weights : an array of weights  

Returns
-------
wmean: Float value indicating weighted mean
"""
def __weighted_mean(row, weights):
    wmean = 0
    for w in range(len(weights)):
        wmean += row['value_'+str(w)]*weights[w]
    return wmean

"""
Internal function to check the inputs for completeness

Parameters
----------
vals    : List of floats/ints
    The original values to adjust
groups  : List of Lists
    A list of groups. Each group will be a list of ints
weights : List of floats
    A list of weights for the groupings.

Returns
-------
None; Throws a ValueError if any of the inputs is incomplete
"""
def __check_individual_inputs(vals, groups, weights):
    # Check values array to see if it is valid
    if vals is None or not len(vals) or list(set(vals)) == [None]:
        raise ValueError('ERR: Values array incomplete ')
    invalid_vals = [val for val in vals if(val is not None and not __is_number(str(val)))]
    if len(invalid_vals):
        raise ValueError('ERR: Values array contains non-numeric values ')
    
    # Check weights and groups array
    if weights is None or None in weights or not len(weights) or \
        groups is None or None in groups or not len(groups):
        raise ValueError('ERR: Weights or groups incomplete ')

    # Check if group length is equal to weights length 
    if len(groups) != len(weights):
        raise ValueError('ERR: Number of groups and weights are unequal')

    # Check if weights array has valid numeric values
    invalid_wts = [wt for wt in weights if(wt is not None and not __is_number(str(wt)))]
    if len(invalid_wts):
        raise ValueError('ERR: Weights array contains non-numeric values ')

    # Check for equal-sized individual groups with associated values
    group_lens = [len(group) for group in groups] + [len(vals)]
    group_lens = list(set(group_lens))
    if(len(group_lens) != 1 or not group_lens[0]):
        raise ValueError('ERR: Unequal or Empty Groups')
    
    # Check if individual groups have valid values e.g, groups should not
    # be an array such as [[ None, None, None], [None, MA, MA]]
    null_groups = [i for i, group in enumerate(groups) if None in group]
    if len(null_groups):
        raise ValueError('ERR: Groups contain None values')

    return

"""
Calculate a group adjustment (demean).
This implementation works for any number of groups.

Parameters
----------
vals    : List of floats/ints
    The original values to adjust
groups  : List of Lists
    A list of groups. Each group will be a list of ints
weights : List of floats
    A list of weights for the groupings.

Returns
-------
A list-like demeaned version of the input values
"""
def group_adjust(vals, groups, weights):
    start = datetime.now()

    __check_individual_inputs(vals, groups, weights)

    # Create a dictionary of group columns to their corresponding values & types
    group_dict  = {}
    group_dtype = {}
    for i in range(len(groups)):
        group_dict[ 'group_'+str(i) ]  = groups[i]
    group_dict[ 'value' ]  = vals

    # Create a dataframe (original) with inputs provided, and replace null values
    group_df = pd.DataFrame(group_dict)
    group_cols = list(group_df.columns)

    # For every granular group level, create an aggregated df.
    # While doing so, also merge it with a final dataframe containing
    # aggregations at all levels
    gparent_df = None
    for i in range(1, len(group_cols)):
        group_by_cols = group_cols[:i]
        gchild_df = group_df[group_by_cols + ['value']].groupby(group_by_cols).mean().reset_index()
        gchild_df = gchild_df.rename(columns = {'value': 'value_'+str(i-1)})
        if gparent_df is not None:
            gparent_df = gparent_df.merge(gchild_df, on = group_by_cols[:-1], how = 'inner' )
        else:
            gparent_df = gchild_df
        
    # Combine the aggregated group df with the original df to get 
    # demeaned values and return 
    gparent_df['mean_value'] = gparent_df.apply(lambda x: __weighted_mean(x, weights), axis = 1)
    group_df = group_df.merge(gparent_df, on = group_cols[:-1], how = 'inner')
    group_df['demean_value'] = group_df['value'] - group_df['mean_value']
    
    end = datetime.now()
    print('Time Taken: {}s'.format((end-start).total_seconds()))
    
    return group_df['demean_value'].values


def test_three_groups():
    print('\nTesting three groups...')
    vals = [1, 2, 3, 8, 5]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'MA', 'MA', 'RI', 'RI']
    grps_3 = ['WEYMOUTH', 'BOSTON', 'BOSTON', 'PROVIDENCE', 'PROVIDENCE']
    weights = [.15, .35, .5]

    adj_vals = group_adjust(vals, [grps_1, grps_2, grps_3], weights)
    # 1 - (USA_mean*.15 + MA_mean * .35 + WEYMOUTH_mean * .5)
    # 2 - (USA_mean*.15 + MA_mean * .35 + BOSTON_mean * .5)
    # 3 - (USA_mean*.15 + MA_mean * .35 + BOSTON_mean * .5)
    # etc ...
    # Plug in the numbers ...
    # 1 - (.15 * 3.8 + .35 * 2.0 + .5 * 1.0) = -0.770
    # 2 - (.15 * 3.8 + .35 * 2.0 + .5 * 2.5) = -0.520
    # 3 - (.15 * 3.8 + .35 * 2.0 + .5 * 2.5) =  0.480
    # etc...

    answer = [-0.770, -0.520, 0.480, 1.905, -1.095]
    print('Actual Output:   {}'.format(answer))
    print('Computed Output: {}'.format(adj_vals))
    for ans, res in zip(answer, adj_vals):
        assert abs(ans - res) < 1e-5

def test_two_groups():
    print('\nTesting two groups...')
    vals = [1, 2, 3, 8, 5]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'RI', 'CT', 'CT', 'CT']
    weights = [.65, .35]

    adj_vals = group_adjust(vals, [grps_1, grps_2], weights)
    # 1 - (.65 * 3.8 + .35 * 1.0) = -1.82
    # 2 - (.65 * 3.8 + .35 * 2.0) = -1.17
    # 3 - (.65 * 3.8 + .35 * 5.33333) = -1.33666
    answer = [-1.82, -1.17, -1.33666, 3.66333, 0.66333]
    print('Actual Output:   {}'.format(answer))
    print('Computed Output: {}'.format(adj_vals))    
    for ans, res in zip(answer, adj_vals):
        assert abs(ans - res) < 1e-5


def test_missing_vals():
    print('\nTesting missing vals...')
    # If you're using NumPy or Pandas, use np.NaN
    # If you're writing pyton, use None
    vals = [1, None, 3, 5, 8, 7]
    # vals = [1, None, 3, 5, 8, 7]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
    weights = [.65, .35]

    adj_vals = group_adjust(vals, [grps_1, grps_2], weights)

    # This should be None or np.NaN depending on your implementation
    # please feel free to change this line to match yours
    answer = [-2.47, np.NaN, -1.170, -0.4533333, 2.54666666, 1.54666666]
    # answer = [-2.47, None, -1.170, -0.4533333, 2.54666666, 1.54666666]
    print('Actual Output:   {}'.format(answer))
    print('Computed Output: {}'.format(adj_vals))
    for ans, res in zip(answer, adj_vals):
        if ans is None:
            assert res is None
        elif np.isnan(ans):
            assert np.isnan(res)
        else:
            assert abs(ans - res) < 1e-5


def test_weights_len_equals_group_len():
    print('\nTesting weights len == Groups len...')
    # Need to have 1 weight for each group

    # vals = [1, np.NaN, 3, 5, 8, 7]
    vals = [1, None, 3, 5, 8, 7]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
    weights = [.65]

    with pytest.raises(ValueError):
        group_adjust(vals, [grps_1, grps_2], weights)


def test_group_len_equals_vals_len():
    print('\nTesting Individual Group len == Vals len ...')
    # The groups need to be same shape as vals
    vals = [1, None, 3, 5, 8, 7]
    grps_1 = ['USA']
    grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
    weights = [.65, .35]

    with pytest.raises(ValueError):
        group_adjust(vals, [grps_1, grps_2], weights)


def test_group_contains_null_vals():
    print('\nTesting if groups contain None values ...')
    # The groups need to be same shape as vals
    vals = [1, None, 3, 5, 8]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'RI', 'RI', None, None]
    weights = [.65, .35]

    with pytest.raises(ValueError):
        group_adjust(vals, [grps_1, grps_2], weights)


def test_only_nulls_in_vals():
    print('\nTesting only nulls in inputs ...')
    vals = [None, None, None, None, None]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'MA', 'MA', 'RI', 'RI']
    grps_3 = ['WEYMOUTH', 'BOSTON', 'BOSTON', 'PROVIDENCE', 'PROVIDENCE']
    weights = [.15, .35, .5]

    with pytest.raises(ValueError):
        group_adjust(vals, [grps_1, grps_2], weights)

def test_non_numeric_vals():
    print('\nTesting non-numeric elements in values array ...')
    vals = [1, 2, 'ab', 4, None]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'MA', 'MA', 'RI', 'RI']
    grps_3 = ['WEYMOUTH', 'BOSTON', 'BOSTON', 'PROVIDENCE', 'PROVIDENCE']
    weights = [.15, .35, .5]

    with pytest.raises(ValueError):
        group_adjust(vals, [grps_1, grps_2], weights)

def test_non_numeric_weights():
    print('\nTesting non-numeric elements in weights array ...')
    vals = [1, 2, 3, 4, None]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'MA', 'MA', 'RI', 'RI']
    grps_3 = ['WEYMOUTH', 'BOSTON', 'BOSTON', 'PROVIDENCE', 'PROVIDENCE']
    weights = [.15, 'a', .5]

    with pytest.raises(ValueError):
        group_adjust(vals, [grps_1, grps_2], weights)


def test_performance():
    print('\nTesting Performance...')
    # vals = 1000000*[1, None, 3, 5, 8, 7]
    # If you're doing numpy, use the np.NaN instead
    vals = 1000000 * [1, np.NaN, 3, 5, 8, 7]
    grps_1 = 1000000 * [1, 1, 1, 1, 1, 1]
    grps_2 = 1000000 * [1, 1, 1, 1, 2, 2]
    grps_3 = 1000000 * [1, 2, 2, 3, 4, 5]
    weights = [.20, .30, .50]

    start = datetime.now()
    group_adjust(vals, [grps_1, grps_2, grps_3], weights)
    end = datetime.now()
    diff = end - start


if __name__ == '__main__':
    # Tests
    test_three_groups()
    test_two_groups()
    test_missing_vals()
    test_weights_len_equals_group_len()
    test_group_len_equals_vals_len()
    test_only_nulls_in_vals()
    test_group_contains_null_vals()
    test_non_numeric_vals()
    test_non_numeric_weights()
    test_performance()
