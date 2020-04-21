# group_adjust_code
Group Adjust Code

This code computes the weighted demeaned values for a given set of hierarchical values.

Functions:

1. Internal Functions:

  __is_number(str): Check if the given str is a number and return True else False

  __weighted_mean(row, weights): given a dictionary containing keys [value_0, value_1, ... ] and a list of weights, computes the weighted mean

  __check_individual_inputs(vals, groups, weights): Checks if the inputs to the group_adjust() function are complete and valid

2. External Functions:
group_adjust(vals, groups, weights): Given values, groups and weights, computes the demeaned values by group hierarchy.
