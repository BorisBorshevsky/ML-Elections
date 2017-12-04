# ML-Elections
Machine Learning 101

# Steps of work
1. Data Cleansing
    1. Missing Data
        1. fill missing columns with relevant values
            - median for numeric fields
            - most common for label for categorial fields
        2. (optional) look for linear correlation with other feature.
        3. (optional) create boolean feature for missing values.
        4. (optional) closest fit (WHAT??)
    2. Noisy Data
        1. Outlier Detection
            1. Nearest Neighbours - remove by choosing outlier.
    3. Data transformation.
        1. Change categories to boolean (0/1) columns.
        2. Change boolean columns to binary.
        3. (Optional) Create categories by grouping. (linear - age, Logarithmic - num of employees )
        4. Scaling
            1. linear ( Xi / max(X) )
        5. (Optional) balance data
            1. if we have one label with way more occurrences than other we should scale it.
2. Feature selection
    
    