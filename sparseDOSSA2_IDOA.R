library(SparseDOSSA2)
library(magrittr)
library(dplyr)
library(ggplot2)
library(readxl)
library(dplyr)

# Load data
genus_data <- read_excel("C:/Users/shaya/OneDrive/Desktop/IDOA/HMP_genus.xlsx",
                          col_names = FALSE)
# Prepare data
genus_data <- genus_data[-1, ]
genus_data <- genus_data[, -1]
genus_data <- t(genus_data)
genus_data = matrix(as.numeric(as.character(genus_data)), ncol=ncol(genus_data))
genus_data <- t(genus_data)
dim(genus_data)

add_row_to_normalize_matrix <- function(mat) {
  # Calculate the sum of each column
  col_sums <- colSums(mat)
  
  # Calculate the values needed to make each column sum to 1
  additional_row <- 1 - col_sums
  
  # Add the new row to the matrix
  new_mat <- rbind(mat, additional_row)
  
  #Reset row names to default numbering
  rownames(new_mat) <- NULL
  
  return(new_mat)
}

genus_data_modified <- add_row_to_normalize_matrix(genus_data)

col_sums <- colSums(genus_data_modified)

library(writexl)

write_xlsx(as.data.frame(t(genus_data_modified)),
           path = "C:/Users/shaya/OneDrive/Desktop/IDOA/genus_data_ordered_modified.xlsx")

# fit spareDOSSA2
fitted <- fit_SparseDOSSA2(data = genus_data_modified,
                           control = list(verbose = TRUE))

genus_data_simulation <- SparseDOSSA2(template = fitted, 
                                      n_sample = 1000,
                                      new_features = FALSE,
                                      verbose = FALSE)

genus_data_simulated_samples <- genus_data_simulation$simulated_data
genus_data_simulated_samples <- as.data.frame(genus_data_simulated_samples)
filtered_species_genus <- genus_data_simulation$template$l_filtering$ind_feature
filtered_species_genus <- as.data.frame(filtered_species_genus)
filtered_species_genus_binary <- lapply(filtered_species_genus,
                                        as.numeric)
filtered_species_genus_binary <- as.data.frame(filtered_species_genus_binary)

write_xlsx(genus_data_simulated_samples,
           path = "C:/Users/shaya/OneDrive/Desktop/IDOA/genus_data_simulated_samples_modified.xlsx")

write_xlsx(filtered_species_genus_binary,
           path = "C:/Users/shaya/OneDrive/Desktop/IDOA/filtered_species_genus_binary_modified.xlsx")