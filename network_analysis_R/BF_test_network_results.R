# setwd("/home/mje/Projects/Hyp_MEG_MNE/network_analysis_R")
setwd("/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/network_connect_res")



# Libraries ----------------------------------------------------------------
library(dplyr)
library(BayesFactor)

# PRESS load data and run permutation tests--------------------------------------

measures <- c("eff", "deg", "trans")
bands = c("theta", "alpha", "beta", "gamma_low", "gamma_high")
atlases <- c("BA")

colum_names <- c("atlas", "measure", "band", "BF",
                 "mean_normal", "mean_hyp", "sd_normal", "sd_hyp")

counter <- 1


for (atlas in atlases){
    for (measure in measures){
        for (band in bands){
            normal_cond = read.csv(paste(measure, "_press_normal_",
                                         atlas, "_",
                                         band, "_MTC_MNE.csv", sep=""),
                                   header = FALSE)
            hyp_cond = read.csv(paste(measure, "_press_hyp_",
                                      atlas, "_",
                                      band, "_MTC_MNE.csv", sep=""),
                                header = FALSE)

            normal_cond <- as.matrix(normal_cond)
            hyp_cond <- as.matrix(hyp_cond)

            result <- ttestBF(normal_cond, hyp_cond)

            result_name <- paste(atlas, "_", measure, "_", band, sep = "")
            assign(result_name, result)

            if (counter == 1){
                df <- data.frame(atlas,
                                 measure,
                                 band,
                                 exp(result@bayesFactor$bf),
                                 mean(normal_cond),
                                 mean(hyp_cond),
                                 sd(normal_cond),
                                 sd(hyp_cond))
                counter <- 2}
            else {
                df <- rbind(df, data.frame(atlas,
                                           measure,
                                           band,
                                           exp(result@bayesFactor$bf),
                                           mean(normal_cond),
                                           mean(hyp_cond),
                                           sd(normal_cond),
                                           sd(hyp_cond)))
            }
        }
    }
}

names(df) <- colum_names
result_df <- tbl_df(df)
result_df$condition = "Press"

# TONE load data and run permutation tests--------------------------------------

measures <- c("eff", "deg", "trans")
bands = c("theta", "alpha", "beta", "gamma_low", "gamma_high")
atlases <- c("BA")

colum_names <- c("atlas", "measure", "band", "BF",
                 "mean_normal", "mean_hyp", "sd_normal", "sd_hyp")
result_table <- NULL

counter <- 1


for (atlas in atlases){
    for (measure in measures){
        for (band in bands){
            normal_cond = read.csv(paste(measure, "_tone_normal_",
                                         atlas, "_",
                                         band, "_MTC_MNE.csv", sep=""),
                                   header = FALSE)
            hyp_cond = read.csv(paste(measure, "_tone_hyp_",
                                      atlas, "_",
                                      band, "_MTC_MNE.csv", sep=""),
                                header = FALSE)

            normal_cond <- as.matrix(normal_cond)
            hyp_cond <- as.matrix(hyp_cond)

            result <- ttestBF(normal_cond, hyp_cond)

            result_name <- paste(atlas, "_", measure, "_", band, sep = "")
            assign(result_name, result)

            if (counter == 1){
                df <- data.frame(atlas,
                                 measure,
                                 band,
                                 exp(result@bayesFactor$bf),
                                 mean(normal_cond),
                                 mean(hyp_cond),
                                 sd(normal_cond),
                                 sd(hyp_cond))
                counter <- 2}
            else {
                df <- rbind(df, data.frame(atlas,
                                           measure,
                                           band,
                                           exp(result@bayesFactor$bf),
                                           mean(normal_cond),
                                           mean(hyp_cond),
                                           sd(normal_cond),
                                           sd(hyp_cond)))
            }
        }
    }
}

names(df) <- colum_names
result_tone_df <- tbl_df(df)
result_tone_df$condition = "Tone"



# Join and all results ----------------------------------------------------

# setwd("~/mnt/Hyp_meg/result")
result_table <- bind_rows(result_df, result_tone_df)
# write.csv(result_table, "results_deg_eff_trans_tone_press.csv")

