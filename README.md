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
    
    
            
            
            Filling Avg_monthly_expense_when_under_age_21 with Avg_Satisfaction_with_previous_vote due to correlation
Filling Garden_sqr_meter_per_person_in_residancy_area with Avg_monthly_expense_on_pets_or_plants due to correlation
Filling Garden_sqr_meter_per_person_in_residancy_area with Phone_minutes_10_years due to correlation
Filling Yearly_IncomeK with Avg_monthly_household_cost due to correlation
Filling Yearly_IncomeK with Avg_size_per_room due to correlation
Filling Avg_monthly_expense_on_pets_or_plants with Phone_minutes_10_years due to correlation
Filling Avg_monthly_household_cost with Political_interest_Total_Score due to correlation
Before outliar detacction: 10000
After outliar detacction: 10000
Removing features with low variance -   []
RFE - Optimal number of features : 19
RFE - Choosing feature: Avg_monthly_expense_when_under_age_21
RFE - Choosing feature: Avg_Satisfaction_with_previous_vote
RFE - Choosing feature: Looking_at_poles_results
RFE - Choosing feature: Garden_sqr_meter_per_person_in_residancy_area
RFE - Choosing feature: Married
RFE - Choosing feature: Yearly_IncomeK
RFE - Choosing feature: Avg_monthly_expense_on_pets_or_plants
RFE - Choosing feature: Avg_monthly_household_cost
RFE - Choosing feature: Will_vote_only_large_party
RFE - Choosing feature: Phone_minutes_10_years
RFE - Choosing feature: Avg_size_per_room
RFE - Choosing feature: Weighted_education_rank
RFE - Choosing feature: Last_school_grades
RFE - Choosing feature: Political_interest_Total_Score
RFE - Choosing feature: Number_of_valued_Kneset_members
RFE - Choosing feature: Overall_happiness_score
RFE - Choosing feature: Is_Most_Important_Issue_Military
RFE - Choosing feature: Is_Most_Important_Issue_Other
RFE - Choosing feature: Is_Most_Important_Issue_Foreign_Affairs
RFE stratified_k - Optimal number of features : 14
RFE stratified_k - Choosing feature: Garden_sqr_meter_per_person_in_residancy_area
RFE stratified_k - Choosing feature: Married
RFE stratified_k - Choosing feature: Yearly_IncomeK
RFE stratified_k - Choosing feature: Avg_monthly_expense_on_pets_or_plants
RFE stratified_k - Choosing feature: Avg_monthly_household_cost
RFE stratified_k - Choosing feature: Will_vote_only_large_party
RFE stratified_k - Choosing feature: Phone_minutes_10_years
RFE stratified_k - Choosing feature: Avg_size_per_room
RFE stratified_k - Choosing feature: Weighted_education_rank
RFE stratified_k - Choosing feature: Last_school_grades
RFE stratified_k - Choosing feature: Political_interest_Total_Score
RFE stratified_k - Choosing feature: Number_of_valued_Kneset_members
RFE stratified_k - Choosing feature: Overall_happiness_score
RFE stratified_k - Choosing feature: Is_Most_Important_Issue_Foreign_Affairs
MI - Choosing feature: Looking_at_poles_results
MI - Choosing feature: Garden_sqr_meter_per_person_in_residancy_area
MI - Choosing feature: Yearly_IncomeK
MI - Choosing feature: Avg_monthly_expense_on_pets_or_plants
MI - Choosing feature: Avg_monthly_household_cost
MI - Choosing feature: Will_vote_only_large_party
MI - Choosing feature: Phone_minutes_10_years
MI - Choosing feature: Avg_size_per_room
MI - Choosing feature: Weighted_education_rank
MI - Choosing feature: Last_school_grades
MI - Choosing feature: Political_interest_Total_Score
MI - Choosing feature: Number_of_valued_Kneset_members
MI - Choosing feature: Overall_happiness_score
f-classif - Choosing feature: Looking_at_poles_results
f-classif - Choosing feature: Yearly_IncomeK
f-classif - Choosing feature: Avg_monthly_household_cost
f-classif - Choosing feature: Will_vote_only_large_party
f-classif - Choosing feature: Avg_size_per_room
f-classif - Choosing feature: Weighted_education_rank
f-classif - Choosing feature: Last_school_grades
f-classif - Choosing feature: Political_interest_Total_Score
f-classif - Choosing feature: Number_of_valued_Kneset_members
f-classif - Choosing feature: Overall_happiness_score
f-classif - Choosing feature: Is_Most_Important_Issue_Military
f-classif - Choosing feature: Is_Most_Important_Issue_Other
f-classif - Choosing feature: Is_Most_Important_Issue_Foreign_Affairs
Tree Clasifier - Choosing feature: Garden_sqr_meter_per_person_in_residancy_area
Tree Clasifier - Choosing feature: Yearly_IncomeK
Tree Clasifier - Choosing feature: Avg_monthly_expense_on_pets_or_plants
Tree Clasifier - Choosing feature: Avg_monthly_household_cost
Tree Clasifier - Choosing feature: Will_vote_only_large_party
Tree Clasifier - Choosing feature: Phone_minutes_10_years
Tree Clasifier - Choosing feature: Avg_size_per_room
Tree Clasifier - Choosing feature: Weighted_education_rank
Tree Clasifier - Choosing feature: Political_interest_Total_Score
Tree Clasifier - Choosing feature: Number_of_valued_Kneset_members
Tree Clasifier - Choosing feature: Overall_happiness_score
Tree Clasifier - Choosing feature: Is_Most_Important_Issue_Military
Tree Clasifier - Choosing feature: Is_Most_Important_Issue_Foreign_Affairs
### Useful features ###
['Political_interest_Total_Score', 'Avg_monthly_household_cost', 'Number_of_valued_Kneset_members', 'Is_Most_Important_Issue_Other', 'Yearly_IncomeK', 'Is_Most_Important_Issue_Military', 'Married', 'Last_school_grades', 'Looking_at_poles_results', 'Garden_sqr_meter_per_person_in_residancy_area', 'Will_vote_only_large_party', 'Weighted_education_rank', 'Avg_monthly_expense_when_under_age_21', 'Avg_size_per_room', 'Avg_Satisfaction_with_previous_vote', 'Overall_happiness_score', 'Phone_minutes_10_years', 'Is_Most_Important_Issue_Foreign_Affairs', 'Avg_monthly_expense_on_pets_or_plants']
### Redundant features ###
['Avg_monthly_expense_when_under_age_21', 'Garden_sqr_meter_per_person_in_residancy_area', 'Garden_sqr_meter_per_person_in_residancy_area', 'Yearly_IncomeK', 'Yearly_IncomeK', 'Avg_monthly_expense_on_pets_or_plants', 'Avg_monthly_household_cost']
### Final features ###
['Number_of_valued_Kneset_members', 'Is_Most_Important_Issue_Other', 'Married', 'Will_vote_only_large_party', 'Phone_minutes_10_years', 'Is_Most_Important_Issue_Foreign_Affairs', 'Political_interest_Total_Score', 'Last_school_grades', 'Is_Most_Important_Issue_Military', 'Avg_Satisfaction_with_previous_vote', 'Looking_at_poles_results', 'Weighted_education_rank', 'Avg_size_per_room', 'Overall_happiness_score']

          
            
            
Filling Avg_monthly_expense_when_under_age_21 with Avg_Satisfaction_with_previous_vote due to correlation
Filling Garden_sqr_meter_per_person_in_residancy_area with Avg_monthly_expense_on_pets_or_plants due to correlation
Filling Garden_sqr_meter_per_person_in_residancy_area with Phone_minutes_10_years due to correlation
Filling Yearly_IncomeK with Avg_monthly_household_cost due to correlation
Filling Yearly_IncomeK with Avg_size_per_room due to correlation
Filling Avg_monthly_expense_on_pets_or_plants with Phone_minutes_10_years due to correlation
Filling Avg_monthly_household_cost with Political_interest_Total_Score due to correlation
Before outliar detacction: 10000
After outliar detacction: 10000
Removing features with low variance -   []
RFE - Optimal number of features : 14
RFE - Choosing feature: Avg_Satisfaction_with_previous_vote
RFE - Choosing feature: Looking_at_poles_results
RFE - Choosing feature: Married
RFE - Choosing feature: Will_vote_only_large_party
RFE - Choosing feature: Phone_minutes_10_years
RFE - Choosing feature: Avg_size_per_room
RFE - Choosing feature: Weighted_education_rank
RFE - Choosing feature: Last_school_grades
RFE - Choosing feature: Political_interest_Total_Score
RFE - Choosing feature: Number_of_valued_Kneset_members
RFE - Choosing feature: Overall_happiness_score
RFE - Choosing feature: Is_Most_Important_Issue_Military
RFE - Choosing feature: Is_Most_Important_Issue_Other
RFE - Choosing feature: Is_Most_Important_Issue_Foreign_Affairs
RFE stratified_k - Optimal number of features : 13
RFE stratified_k - Choosing feature: Avg_Satisfaction_with_previous_vote
RFE stratified_k - Choosing feature: Married
RFE stratified_k - Choosing feature: Will_vote_only_large_party
RFE stratified_k - Choosing feature: Phone_minutes_10_years
RFE stratified_k - Choosing feature: Avg_size_per_room
RFE stratified_k - Choosing feature: Weighted_education_rank
RFE stratified_k - Choosing feature: Last_school_grades
RFE stratified_k - Choosing feature: Political_interest_Total_Score
RFE stratified_k - Choosing feature: Number_of_valued_Kneset_members
RFE stratified_k - Choosing feature: Overall_happiness_score
RFE stratified_k - Choosing feature: Is_Most_Important_Issue_Military
RFE stratified_k - Choosing feature: Is_Most_Important_Issue_Other
RFE stratified_k - Choosing feature: Is_Most_Important_Issue_Foreign_Affairs
MI - Choosing feature: AVG_lottary_expanses
MI - Choosing feature: Avg_Satisfaction_with_previous_vote
MI - Choosing feature: Looking_at_poles_results
MI - Choosing feature: Will_vote_only_large_party
MI - Choosing feature: Phone_minutes_10_years
MI - Choosing feature: Avg_size_per_room
MI - Choosing feature: Weighted_education_rank
MI - Choosing feature: Last_school_grades
MI - Choosing feature: Political_interest_Total_Score
MI - Choosing feature: Number_of_valued_Kneset_members
MI - Choosing feature: Overall_happiness_score
MI - Choosing feature: Is_Most_Important_Issue_Military
f-classif - Choosing feature: Looking_at_poles_results
f-classif - Choosing feature: Will_vote_only_large_party
f-classif - Choosing feature: Phone_minutes_10_years
f-classif - Choosing feature: Avg_size_per_room
f-classif - Choosing feature: Weighted_education_rank
f-classif - Choosing feature: Last_school_grades
f-classif - Choosing feature: Political_interest_Total_Score
f-classif - Choosing feature: Number_of_valued_Kneset_members
f-classif - Choosing feature: Overall_happiness_score
f-classif - Choosing feature: Is_Most_Important_Issue_Military
f-classif - Choosing feature: Is_Most_Important_Issue_Other
f-classif - Choosing feature: Is_Most_Important_Issue_Foreign_Affairs
Tree Clasifier - Choosing feature: Married
Tree Clasifier - Choosing feature: Will_vote_only_large_party
Tree Clasifier - Choosing feature: Phone_minutes_10_years
Tree Clasifier - Choosing feature: Avg_size_per_room
Tree Clasifier - Choosing feature: Weighted_education_rank
Tree Clasifier - Choosing feature: Last_school_grades
Tree Clasifier - Choosing feature: Political_interest_Total_Score
Tree Clasifier - Choosing feature: Number_of_valued_Kneset_members
Tree Clasifier - Choosing feature: Overall_happiness_score
Tree Clasifier - Choosing feature: Is_Most_Important_Issue_Military
Tree Clasifier - Choosing feature: Is_Most_Important_Issue_Other
Tree Clasifier - Choosing feature: Is_Most_Important_Issue_Foreign_Affairs
### Useful features ###
['Will_vote_only_large_party', 'Political_interest_Total_Score', 'Number_of_valued_Kneset_members', 'Is_Most_Important_Issue_Other', 'Overall_happiness_score', 'Married', 'Last_school_grades', 'Weighted_education_rank', 'Avg_Satisfaction_with_previous_vote', 'Avg_size_per_room', 'Looking_at_poles_results', 'Is_Most_Important_Issue_Military', 'Phone_minutes_10_years', 'Is_Most_Important_Issue_Foreign_Affairs', 'AVG_lottary_expanses']
### Redundant features ###
['Avg_monthly_expense_when_under_age_21', 'Garden_sqr_meter_per_person_in_residancy_area', 'Garden_sqr_meter_per_person_in_residancy_area', 'Yearly_IncomeK', 'Yearly_IncomeK', 'Avg_monthly_expense_on_pets_or_plants', 'Avg_monthly_household_cost']
### Final features ###
['Political_interest_Total_Score', 'Married', 'Is_Most_Important_Issue_Other', 'Overall_happiness_score', 'Number_of_valued_Kneset_members', 'Last_school_grades', 'Weighted_education_rank', 'Avg_Satisfaction_with_previous_vote', 'Will_vote_only_large_party', 'Looking_at_poles_results', 'Avg_size_per_room', 'Is_Most_Important_Issue_Military', 'Phone_minutes_10_years', 'Is_Most_Important_Issue_Foreign_Affairs', 'AVG_lottary_expanses']
            