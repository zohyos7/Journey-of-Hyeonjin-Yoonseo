setwd("~/Journey-of-Hyeonjin-Yoonseo/")

rm(list=ls())

# Simulation parameters
seed <- 0777    # do not change the seed number
num_subjs <- 26 # number of subjects
num_trials <- 108 # number of trials per subject per advisor
n_advisors <- 3
advisor_1_correct <- 0.25  # reward probability in option 1
advisor_2_correct <- 0.50  # reward probability in option 2
advisor_3_correct <- 0.75

# Set seed
set.seed(seed)   # always set a seed number for this homework!

# Generated True parameters 
simul_pars <- data.frame(Eta = rnorm(N, 0.20, 0.15),
                         Beta = rnorm(N, 9.00, 8.00),
                         subjID  = 1:num_subjs)

# For storing simulated choice data for all subjects
# all_data <- NULL

tmp_data_choice = array(-1, c(26, 3, 36))
tmp_data_advisor = array(-1, c(26, 3, 36))

for (i in 1:N){ 
  # Individual-level (i.e. per subject) parameter values
  Eta <- simul_pars$Eta[i]
  Beta <- simul_pars$Beta[i]
  
  # initialize some variables
  
  for (a in 1:n_advisor) {
    
    V = 0
  
    for (t in 1:T)  {
      
      # outcome
      outcome = rbinom(size=1, n = 1, prob = pr_correct_advisor[a])
      if (outcome == 0){outcome = -1}
      # after receiving outcome (feedback), update sv[t+1]
      # prediction error (PE)
      PE = outcome - V
    
      # update stimulus value (sv) of the chosen option
      V = V + Eta * PE
      
      # Prob of choosing option 2
      prob_choose = 1 / (1 + exp(- Beta * V))  # exploration/exploitation parameter is set to 1
      
      # choice
      choice = rbinom(size=1, n = 1, prob = prob_choose )
      # choice = choice + 1  # 0 or 1 --> 1 (option 1) or 2 (option 2)
      
      # append simulated task/response to subject data
      tmp_data_choice[i,a,t] = choice
      tmp_data_advisor[i,a,t] = outcome
    } # end of t loop
  }
}  

save(simul_pars,tmp_data_advisor, tmp_data_choice, file = 'tmp_data_simul2.rdata')

# Write out data
#write.table(all_data, file = "simul_data_hw5_num200.txt", row.names = F, col.names = T, sep = "\t")
