library(readxl)

cohort <- read_excel("C:/Users/shaya/OneDrive/Desktop/IDOA/HMP_cohorts/Saliva.xlsx",
                      col_names = FALSE)

# Normalize cohort.

col_sums <- colSums(cohort)
norm_cohort <- cohort / matrix(rep(col_sums, each = nrow(cohort)), nrow = nrow(cohort))

# Calculate the mean of each row
row_means <- apply(norm_cohort, 1, mean)

# Identify rows with mean greater than or equal to 0.0005
rows_to_keep <- row_means >= 0.0005

# Subset the matrix to keep only those rows
filtered_cohort <- norm_cohort[rows_to_keep, ]

# Normalize filtered_cohort
filtered_cohort <- filtered_cohort / matrix(rep(col_sums, each = nrow(filtered_cohort)),
                                            nrow = nrow(filtered_cohort))

# Fit MIDAS
fit <- MIDAS::Midas.setup(t(filtered_cohort), n.break.ties = 100, fit.beta = F)

simulation <- MIDAS::Midas.modify(fit,
                                  mean.rel.abund = NULL, 
                                  taxa.1.prop = NULL)

Midas_simulated_cohort <- MIDAS::Midas.sim(simulation, only.rel=T)

Midas_simulated_cohort.rel_abund = Midas_simulated_cohort$sim_rel

write_xlsx(as.data.frame(Midas_simulated_cohort.rel_abund),
           path = "C:/Users/shaya/OneDrive/Desktop/IDOA/Midas_saliva.xlsx")
