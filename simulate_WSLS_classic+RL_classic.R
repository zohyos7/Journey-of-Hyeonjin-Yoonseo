setwd("~/Journey-of-Hyeonjin-Yoonseo/")

rm(list=ls())

# Simulation parameters
seed <- 0777    # do not change the seed number!
num_subjs  <- 26  # number of subjects
num_trials <- 108 # number of trials per subject
num_advisors <- 3
advisor_1_correct <- 0.25  # reward probability in option 1
advisor_2_correct <- 0.50  # reward probability in option 2
advisor_3_correct <- 0.75

# Set seed
set.seed(seed)   # always set a seed number for this homework!

?rnorm

#Generated True parameters 
{
simul_pars <- data.frame(Eta = rnorm(num_subjs, 0.30, 0.15) ,
                         Beta = rnorm(num_subjs, 5.50, 5.00),
                         pSW = rnorm(num_subjs, 0.80, 0.20),
                         pSL = rnorm(num_subjs, 0.30, 0.15),
                         K = rnorm(num_subjs, 0.5, 0.3),
                         subjID  = 1:num_subjs)

simul_pars$Eta[simul_pars$Eta < 0 | simul_pars$Eta > 1 ] = 0.30
simul_pars$Beta[simul_pars$Beta < 0 ] = 5.5
simul_pars$pSW[simul_pars$pSW < 0 | simul_pars$pSW > 1] = 0.80
simul_pars$pSL[simul_pars$pSL < 0 | simul_pars$pSL > 1] = 0.30
simul_pars$K[simul_pars$K < 0 | simul_pars$K > 1] = 0.5
}

#'True' parameters estimated from the data
{
parameters <- rstan::extract(output6)

Etamean <- vector()
Betamean <- vector()
pSWmean <- vector()
pSLmean <- vector()
Kmean <- vector()

for(i in 1:26){Etamean[i] <- mean(parameters$Eta[,i])}
for(i in 1:26){Betamean[i] <- mean(parameters$Beta[,i])}
for(i in 1:26){pSWmean[i] <- mean(parameters$pSW[,i])}
for(i in 1:26){pSLmean[i] <- mean(parameters$pSL[,i])}
for(i in 1:26){Kmean[i] <- mean(parameters$K[,i])}

simul_pars <- data.frame(Etamean,
                         Betamean,
                         pSWmean,
                         pSLmean,
                         Kmean,
                         subjID = 1:26)

}

# For storing simulated choice data for all subjects
Choice <- array(-1, c(num_subjs, num_advisors, num_trials))
AdvisorCorrect <- array(-1, c(num_subjs, num_advisors, num_trials))
Outcome <- array(0, c(num_subjs, num_advisors, num_trials))

for (i in 1:num_subjs) {
  # Individual-level (i.e. per subject) parameter values
  Eta <- simul_pars$Eta[i]
  Beta <- simul_pars$Beta[i]
  pSW <- simul_pars$pSW[i]
  pSL <- simul_pars$pSL[i]
  K <- simul_pars$K[i]
  
  AdvisorCorrect[i, 1, ] = rbinom(size = 1, n = num_trials, prob = advisor_1_correct)
  AdvisorCorrect[i, 2, ] = rbinom(size = 1, n = num_trials, prob = advisor_2_correct)
  AdvisorCorrect[i, 3, ] = rbinom(size = 1, n = num_trials, prob = advisor_2_correct)
  Outcome <- AdvisorCorrect
  Outcome[Outcome==0] = -1
  
  for (a in 1:num_advisors) {

    V = 0

    for (t in 1:num_trials) {
      
      pRL = 1 / (1 + exp(-Beta*V))
      
      if (t == 1) {
        Choice[i, a, t] = rbinom(size = 1, n = 1, prob = 0.5*K + pRL*(1-K))
        }
      else {
        if (AdvisorCorrect[i, a, t-1] == Choice[i, a, t-1]) {
          if (Choice[i, a, t-1] == 1) {
            Choice[i, a, t] = rbinom(size = 1, n = 1, prob = pSW*K + pRL*(1-K))
          }
          else {
            Choice[i, a, t] = rbinom(size = 1, n = 1, prob = (1-pSW)*K + pRL*(1-K))
           }
          }
        else {
          if (Choice[i, a, t-1] == 1) {
              Choice[i, a, t] = rbinom(size = 1, n = 1, prob = (1-pSL)*K + pRL*(1-K))
            }
          else{
              Choice[i, a, t] = rbinom(size = 1, n = 1, prob = pSL*K + pRL*(1-K))
            }
          }
        }
      }
      PE = Outcome[i, a, t] - V;
      V = V + Eta*PE
    }
  }



dataListC <- list(
  N = num_subjs,
  T = num_trials*num_advisors,
  A = num_advisors,
  Choice = Choice,
  AdvisorCorrect = AdvisorCorrect,
  Outcome = Outcome
)

output6 = stan("WSLS_classic+RL_classic.stan",
               data = dataListC, pars = c( "Eta", "Beta","pSW","pSL","K","Choice_pred", "log_lik"), 
               iter = 2000, warmup=1000, chains=2, cores=2)
print(output6)


parameters <- rstan::extract(output6)

mean(parameters$Eta)
mean(parameters$Beta)
mean(parameters$pSW)
mean(parameters$pSL)
mean(parameters$K)


Eta_mean = apply(parameters$Eta, 2, mean)
Eta_sd = apply(parameters$Eta, 2, mean)

Beta_mean = apply(parameters$Eta, 2, mean)
Beta_sd = apply(parameters$Eta, 2, mean)

pSW_mean = apply(parameters$pSW, 2, mean)
pSW_sd = apply(parameters$pSW, 2, sd)

pSL_mean = apply(parameters$pSL, 2, mean)
pSL_sd = apply(parameters$pSL, 2, sd)

K_mean = apply(parameters$K, 2, mean)
K_sd = apply(parameters$K, 2, mean)

stan_plot(output6, "pSW", show_density=T)
stan_plot(output6, "pSL", show_density=T)

Eta_comparison <- data.frame(true = simul_pars$Eta, posterior = Eta_mean, posterior_sd = Eta_sd)
Beta_comparison <- data.frame(true = simul_pars$Beta, posterior = Beta_mean, posterior_sd = Beta_sd)
pSW_comparison <- data.frame(true = simul_pars$pSW, posterior = pSW_mean, posterior_sd = pSW_sd)
pSL_comparison <- data.frame(true = simul_pars$pSL, posterior = pSL_mean, posterior_sd = pSL_sd)
K_comparison<- data.frame(true = simul_pars$K, posterior = K_mean, posterior_sd = K_sd)


ggplot(pSW_comparison, aes(x=true,y=posterior)) + geom_point(colour = "blue", size = 2)+
  geom_errorbar(aes(ymax=posterior+posterior_sd,ymin=posterior-posterior_sd,width=0)) +
  geom_abline(intercept=0, slope=1, color = "gray", linetype = "dashed", size = 0.5) +
  ggtitle("pSW")


ggplot(pSL_comparison, aes(x=true,y=posterior)) + geom_point(colour = "blue", size = 2)+
  geom_errorbar(aes(ymax=posterior+posterior_sd,ymin=posterior-posterior_sd,width=0)) +
  geom_abline(intercept=0, slope=1, color = "gray", linetype = "dashed", size = 0.5) +
  ggtitle("pSL")


ggplot(Eta_comparison, aes(x=true,y=posterior)) + geom_point(colour = "blue", size = 2)+
  geom_errorbar(aes(ymax=posterior+posterior_sd,ymin=posterior-posterior_sd,width=0)) +
  geom_abline(intercept=0, slope=1, color = "gray", linetype = "dashed", size = 0.5) +
  ggtitle("Eta")


ggplot(Beta_comparison, aes(x=true,y=posterior)) + geom_point(colour = "blue", size = 2)+
  geom_errorbar(aes(ymax=posterior+posterior_sd,ymin=posterior-posterior_sd,width=0)) +
  geom_abline(intercept=0, slope=1, color = "gray", linetype = "dashed", size = 0.5) +
  ggtitle("Beta")

ggplot(K_comparison, aes(x=true,y=posterior)) + geom_point(colour = "blue", size = 2)+
  geom_errorbar(aes(ymax=posterior+posterior_sd,ymin=posterior-posterior_sd,width=0)) +
  geom_abline(intercept=0, slope=1, color = "gray", linetype = "dashed", size = 0.5) +
  ggtitle("K")


