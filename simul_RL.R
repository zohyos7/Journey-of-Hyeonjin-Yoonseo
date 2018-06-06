setwd("~/Journey-of-Hyeonjin-Yoonseo/")

rm(list=ls())

# Simulation parameters
seed <- 0777    # do not change the seed number
num_subjs <- 26 # number of subjects
num_trials <- 108 # number of trials per subject per advisor
num_advisors <- 3
advisor_1_correct <- 0.25  # reward probability in option 1
advisor_2_correct <- 0.50  # reward probability in option 2
advisor_3_correct <- 0.75

# Set seed
set.seed(seed)   # always set a seed number for this homework!

# Generated True parameters 
{simul_pars <- data.frame(Eta = rnorm(num_subjs, 0.20, 0.15),
                         Beta = rnorm(num_subjs, 9.00, 8.00),
                         subjID  = 1:num_subjs)

simul_pars$Eta[simul_pars$Eta < 0 | simul_pars$Eta > 1] = 0.20
simul_pars$Beta[simul_pars$Beta < 0 | simul_pars$Beta > 1] = 9.00
}

#True parameters with real data
{
parameters <- rstan::extract(output2)
  
Etamean <- vector()
Betamean <- vector()

for(i in 1:26){Etamean[i] <- mean(parameters$Eta[,i])}
for(i in 1:26){Betamean[i] <- mean(parameters$Beta[,i])}

simul_pars <- data.frame(Etamean,
                         Betamean,
                         subjID = 1:26)
}

# For storing simulated choice data for all subjects
# all_data <- NULL

Choice <- array(-1, c(num_subjs, num_advisors, num_trials))
AdvisorCorrect <- array(0, c(num_subjs, num_advisors, num_trials))


for (i in 1:num_subjs){ 
  # Individual-level (i.e. per subject) parameter values
  Eta <- simul_pars$Eta[i]
  Beta <- simul_pars$Beta[i]
  
  AdvisorCorrect[i, 1, ] = rbinom(size = 1, n = 36, prob = advisor_1_correct)
  AdvisorCorrect[i, 2, ] = rbinom(size = 1, n = 36, prob = advisor_2_correct)
  AdvisorCorrect[i, 3, ] = rbinom(size = 1, n = 36, prob = advisor_2_correct)
  Outcome <- AdvisorCorrect
  Outcome[Outcome == 0] = -1
  
  for (a in 1:num_advisors) {
    
    V = 0
  
    for (t in 1:num_trials) {
      
      Choice[i,a,t] = rbinom(size = 1, n = 1, prob = 1/(1 + exp(- Beta * V)))
      
      PE = Outcome[i, a, t] - V
    
      # update stimulus value (sv) of the chosen option
      V = V + Eta * PE
    } # end of t loop
  }
}  

dataList <- list(
  N = num_subjs,
  T = num_trials*num_advisors,
  A = num_advisors,
  Choice = Choice,
  Outcome = Outcome
)


output_h = stan("RL_classic_hierarchy.stan", data = dataList, pars=c("Eta", "Beta"),
              iter = 2000, warmup=1000, chains=2, cores=2)


output_nh = stan("RL_classic.stan", data = dataList, pars=c("Eta", "Beta"),
              iter = 2000, warmup=1000, chains=2, cores=2)

parameters <- rstan::extract(output_nh)

mean(parameters$Eta)
mean(parameters$Beta)

Eta_mean = apply(parameters$Eta, 2, mean)
Eta_sd = apply(parameters$Eta, 2, sd)
Beta_mean = apply(parameters$Beta, 2, mean)
Beta_sd = apply(parameters$Beta, 2, sd)

stan_plot(output_h, "Eta", show_density=T)
stan_plot(output_h, "Beta", show_density=T)

Eta_comparison <- data.frame(true = simul_pars$Eta, posterior = Eta_mean, posterior_sd = Eta_sd)
Beta_comparison <- data.frame(true = simul_pars$Beta, posterior = Beta_mean, posterior_sd = Beta_sd)


ggplot(Eta_comparison, aes(x=true,y=posterior)) + geom_point(colour = "blue", size = 2)+
  geom_errorbar(aes(ymax=posterior+posterior_sd,ymin=posterior-posterior_sd,width=0)) +
  geom_abline(intercept=0, slope=1, color = "gray", linetype = "dashed", size = 0.5) +
  ggtitle("Eta")


ggplot(Beta_comparison, aes(x=true,y=posterior)) + geom_point(colour = "blue", size = 2)+
  geom_errorbar(aes(ymax=posterior+posterior_sd,ymin=posterior-posterior_sd,width=0)) +
  geom_abline(intercept=0, slope=1, color = "gray", linetype = "dashed", size = 0.5) +
  ggtitle("Beta")






