library(SparseDOSSA2)
library(magrittr)
library(dplyr)
library(ggplot2)
library(readxl)
library(dplyr)

# Load data
cohort <- read_excel("C:/Users/Shaya/Desktop/IDOA_project/HMP_cohorts/Saliva.xlsx",
                     col_names = FALSE)

col_sums <- colSums(cohort)
norm_cohort <- cohort / matrix(rep(col_sums, each = nrow(cohort)), nrow = nrow(cohort))

# Calculate the mean of each row
row_means <- apply(norm_cohort, 1, mean)

# Identify rows with mean greater than or equal to 0.0005
rows_to_keep <- row_means >= 0.0005

# Subset the matrix to keep only those rows
filtered_cohort <- norm_cohort[rows_to_keep, ]

col_sums <- colSums(filtered_cohort)
filtered_cohort <- filtered_cohort / matrix(rep(col_sums, each = nrow(filtered_cohort)),
                                            nrow = nrow(filtered_cohort))

library(writexl)

write_xlsx(as.data.frame(filtered_cohort),
           path = "C:/Users/Shaya/Desktop/HMP_cohorts/Saliva_filtered.xlsx")

fitted <- fit_SparseDOSSA2(data = filtered_cohort,
                           control = list(verbose = TRUE))

simulation <- SparseDOSSA2(template = fitted, 
                           n_sample = 1000,
                           new_features = FALSE,
                           verbose = FALSE)

simulated_samples <- simulation$simulated_data
simulated_samples <- as.data.frame(simulated_samples)
filtered_species <- simulation$template$l_filtering$ind_feature
filtered_species <- as.data.frame(filtered_species)
filtered_species_binary <- lapply(filtered_species, as.numeric)
filtered_species_binary <- as.data.frame(filtered_species_binary)

write_xlsx(simulated_samples,
           path = "C:/Users/Shaya/Desktop/IDOA_project/Simulations data/SparseDOSSA2_simulated_samples_saliva.xlsx")

write_xlsx(filtered_species_binary,
           path = "C:/Users/Shaya/Desktop/IDOA_project/Simulations data/SparseDOSSA2_filtered_species_binary_saliva.xlsx")



