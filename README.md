# group_adjust_code
Group Adjust Code

This code computes the weighted demeaned values for a given set of hierarchical values.

Functions:

1. Internal Functions:

  * __is_number(str): Check if the given str is a number and return True else False

  * __weighted_mean(row, weights): given a dictionary containing keys [value_0, value_1, ... ] and a list of weights, computes the weighted mean

  * __check_individual_inputs(vals, groups, weights): Checks if the inputs to the group_adjust() function are complete and valid

2. External Functions:
  * group_adjust(vals, groups, weights): Given values, groups and weights, computes the demeaned values by group hierarchy.
  
3. Tests:
Following tests are run on the group_adjust function:
  * test_three_groups()
  * test_two_groups()
  * test_missing_vals()
  * test_weights_len_equals_group_len()
  * test_group_len_equals_vals_len()
  * test_only_nulls_in_vals()
  * test_group_contains_null_vals()
  * test_non_numeric_vals()
  * test_non_numeric_weights()
  * test_performance()
