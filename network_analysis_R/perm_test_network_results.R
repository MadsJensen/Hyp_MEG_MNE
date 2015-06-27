# setwd("/home/mje/Projects/Hyp_MEG_MNE/network_analysis_R")
setwd("/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/network_connect_res")



# Libraries ----------------------------------------------------------------
library(dplyr)

# Functions ---------------------------------------------------------------

permutation_test <- function(a, b, number_permutations=5000){
    c = c(a, b)

    # Observed difference
    diff.observed = mean(b) - mean(a)

    diff.random = NULL
    for (i in 1 : number_permutations) {

        # Sample from the combined dataset
        a.random = sample (c, length(a), TRUE)
        b.random = sample (c, length(b), TRUE)

        # Null (permuated) difference
        diff.random[i] = mean(b.random) - mean(a.random)
    }

    # P-value is the fraction of how many times the permuted difference is
    # equal or more extreme than the observed difference
    pvalue = sum(abs(diff.random) >= abs(diff.observed)) / number_permutations
    result <- list(pvalue = pvalue,
                    random = diff.random,
                    obs_diff = diff.observed)
    return(result)
}


# Load data and run permutation tests--------------------------------------

conditions <- c("eff", "deg", "trans")
bands = c("theta", "alpha", "beta", "gamma_low", "gamma_high")
atlases <- c("BA", "DK", "DX")

colum_names <- c("atlas", "condition", "band", "pvalue",
                 "mean_normal", "mean_hyp", "sd_normal", "sd_hyp", "obs_diff")
result_table <- NULL

counter <- 1


for (atlas in atlases){
    for (condition in conditions){
        for (band in bands){
            normal_cond = read.csv(paste(condition, "_press_normal_",
                                         atlas, "_",
                                         band, ".csv", sep=""),
                                   header = FALSE)
            hyp_cond = read.csv(paste(condition, "_press_hyp_DX_",
                                      band, ".csv", sep=""),
                                header = FALSE)

            normal_cond <- as.matrix(normal_cond)
            hyp_cond <- as.matrix(hyp_cond)

            result <- permutation_test(normal_cond, hyp_cond)

            result_name <- paste(atlas, "_", condition, "_", band, sep = "")
            assign(result_name, result)

            if (counter == 1){
                df <- data.frame(atlas,
                                 condition,
                                 band,
                                 result$pvalue,
                                 mean(normal_cond),
                                 mean(hyp_cond),
                                 sd(normal_cond),
                                 sd(hyp_cond),
                                 result$obs_diff)
                counter <- 2}
            else {
                df <- rbind(df, data.frame(atlas,
                                           condition,
                                           band,
                                           result$pvalue,
                                           mean(normal_cond),
                                           mean(hyp_cond),
                                           sd(normal_cond),
                                           sd(hyp_cond),
                                           result$obs_diff))
            }
        }
    }
}

names(df) <- colum_names
result_df <- tbl_df(df)

result_df$fdr_corr <- p.adjust(result_df$pvalue, method = "fdr")
