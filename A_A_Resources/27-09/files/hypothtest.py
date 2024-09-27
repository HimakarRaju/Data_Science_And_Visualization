'''
Start
  |
  v
Input: Sample1, Sample2
  |
  v
Calculate Mean of Sample1 (mean1) and Sample2 (mean2)
  |
  v
Calculate Variance of Sample1 (var1) and Sample2 (var2)
  |
  v
Compute Pooled Standard Deviation (pooled_std)
  |
  v
Calculate t-statistic:
  t = (mean1 - mean2) / (pooled_std * sqrt(1/len(sample1) + 1/len(sample2)))
  |
  v
Output: t-statistic value
  |
  v
End


'''

"""
pooled standard deviation is calculated by getting sqr root of pooled variable.

step1 : x, y
step 2: find mean of sample1 and sample 2
step 3 : calc variances of sample1 and sample2
	variance of sample = sum (x)-mean(sample1)**2 for each value in the sample) / (no. of samples - 1 )
step 4 : pooled variable = (length of sample1) - 1 * (variance of sample 1) + (length of sample2) - 1 * (variance of sample 2) / (length of sample 1 + length of sample 2) -2
	pooled std. division = sqt. of pooled variable 
step 5 : to cal t_statistics = (mean1-mean2)/(pooled std. division * sqrt(1/len(sample1)+1/len(sample2))

"""




import math
def two_sample_t_test(sample1, sample2):
    # Calculate the means
    mean1 = sum(sample1) / len(sample1)
    mean2 = sum(sample2) / len(sample2)

    # Calculate the variances
    var1 = sum((x - mean1) ** 2 for x in sample1) / (len(sample1) - 1)
    var2 = sum((x - mean2) ** 2 for x in sample2) / (len(sample2) - 1)

    # Calculate the pooled standard deviation
    pooled_var = (((len(sample1) - 1) * var1) + ((len(sample2) - 1)
                  * var2)) / (len(sample1) + len(sample2) - 2)

    pooled_std = math.sqrt(pooled_var)

    # Calculate the t-statistic
    t_stat = (mean1 - mean2) / (pooled_std *
                                math.sqrt(1 / len(sample1) + 1 / len(sample2)))

    return t_stat


# Example usage
sample1 = [10, 12, 23, 23, 16, 23, 21, 16]
sample2 = [14, 22, 24, 20, 18, 22, 24, 20]

t_stat = two_sample_t_test(sample1, sample2)
print(f"T-statistic: {t_stat}")
