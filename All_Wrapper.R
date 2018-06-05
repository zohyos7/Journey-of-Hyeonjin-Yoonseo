rm(list=ls())

library(R.matlab)
library(rstan)
library(loo)

setwd("~/Yoonseo Zoh/2018-1/Computational Modeling/Final Project/Codes")

#data
{
  dat <- readMat("AllData.mat")
dat <- dat$AllData

subj <- vector()
col2 <- list()
col3 <- list()
col4 <- list()

for(i in 1:26){
  subj[i] <- sapply(dat[[i]], function(x) sapply(x, function(y) y))
  col2[[i]] <- sapply(dat[[i+26]], function(x) sapply(x, function(y) y))
  col3[[i]] <- sapply(dat[[i+52]], function(x) sapply(x, function(y) y))
  col4[[i]] <- sapply(dat[[i+78]], function(x) sapply(x, function(y) y))
}

AdvisorCorrect_raw <- matrix(nrow=108,ncol=26)
for(i in 1:26) {AdvisorCorrect_raw[,i] <- col3[[i]][[2]][[1]][[16]]}

Advisor <- matrix(nrow=108, ncol=26)
for(i in 1:26) {Advisor[,i] <-col3[[i]][[2]][[1]][[5]]}

Choice_raw <- matrix(nrow=108,ncol=26)
for(i in 1:26) {Choice_raw[,i] <- col3[[i]][[2]][[1]][[3]]}

numSubjs = ncol(AdvisorCorrect_raw)
Trials = nrow(AdvisorCorrect_raw)
A = max(Advisor)

AdvisorCorrect <- array(0, c(numSubjs, A, Trials/A))

for (i in 1:numSubjs){
  AdvisorCorrect[i,,] <- matrix(data=NA, nrow = Trials/A, ncol = A)
  for (j in 1:3){
    AdvisorCorrect[i,j,] <- AdvisorCorrect_raw[,i][Advisor[,i]==j]
  }
}

Choice <- array(-1, c(numSubjs, A, Trials/A))

for (i in 1:numSubjs){
  Choice[i,,] <- matrix(data=NA, nrow = Trials/A, ncol = A)
  for (j in 1:3){
    Choice[i,j,] <- Choice_raw[,i][Advisor[,i]==j]
  }
}

Choice[is.nan(Choice)] = -1
Outcome = AdvisorCorrect
Outcome[Outcome==0] = -1

rm(col2, col3, col4, dat, j, i, subj)


dataListA <- list(
  N = numSubjs,
  T = Trials,
  A = A,
  Choice = Choice,
  AdvisorCorrect = AdvisorCorrect
)

dataListB <- list(
  N = numSubjs,
  T = Trials,
  A = A,
  Choice = Choice,
  Outcome = Outcome
)

dataListC <- list(
  N = numSubjs,
  T = Trials,
  A = A,
  Choice = Choice,
  AdvisorCorrect = AdvisorCorrect,
  Outcome = Outcome
)
}

#WSLS(classic)

output1 = stan("WSLS_classic.stan", data = dataListA, pars = c("pSW","pSL","Choice_pred","log_lik"),
              iter = 2000, warmup=1000, chains=2, cores=2)

#RL(classic)

output2 = stan("RL_classic.stan", data = dataListB, pars = c("Eta","Beta","Choice_pred","log_lik"),
                         iter = 2000, warmup=1000, chains=2, cores=2)

#WSLS_learning

output3 = stan("WSLS_learning.stan", data = dataListA, 
                         pars = c("theta_pSW","theta_pSL","init_pSW","init_pSL","Choice_pred","log_lik"),
                         iter = 2000, warmup=1000, chains=2, cores=2)

#RL_transition1

output4 = stan("transition_RL_v2.stan", data = dataListA, pars = c("Eta", "Beta","Choice_pred","log_lik"),
               iter = 2000, warmup=1000, chains=2, cores=2)


#RL_transition2

output5 = stan("transition_RL_v2.stan", data = dataListA, pars = c("Eta", "Beta","Choice_pred","log_lik"),
               iter = 2000, warmup=1000, chains=2, cores=2)

#WSLS_classic + RL_classic

output6 = stan("WSLS_classic+RL_classic.stan",
              data = dataListC, pars = c( "Eta", "Beta","pSW","pSL","K","Choice_pred", "log_lik"), 
              iter = 2000, warmup=1000, chains=2, cores=2)

# WSLS_learning + RL_classic

output7 = stan("WSLS_learning+RL.stan", 
              data = dataListC, pars = c( "Eta", "Beta",'theta_pSW','theta_pSL','init_pSW','init_pSL','K','Choice_pred', 'log_lik'), 
              iter = 2000, warmup=1000, chains=2, cores=2)


lik_1 <- loo::extract_log_lik(output1, parameter_name = 'log_lik')
lik_2 <- loo::extract_log_lik(output2, parameter_name = 'log_lik')
lik_3 <- loo::extract_log_lik(output3, parameter_name = 'log_lik')
lik_4 <- loo::extract_log_lik(output4, parameter_name = 'log_lik')
lik_5 <- loo::extract_log_lik(output5, parameter_name = 'log_lik')
lik_6 <- loo::extract_log_lik(output6, parameter_name = 'log_lik')
lik_7 <- loo::extract_log_lik(output7, parameter_name = 'log_lik')


LOOIC_1 <- loo::loo(lik_1, cores =2)
LOOIC_2 <- loo::loo(lik_2, cores =2)
LOOIC_3 <- loo::loo(lik_3, cores =2)
LOOIC_4 <- loo::loo(lik_4, cores =2)
LOOIC_5 <- loo::loo(lik_5, cores =2)
LOOIC_6 <- loo::loo(lik_6, cores =2)
LOOIC_7 <- loo::loo(lik_7, cores =2)


library(loo)
LOOIC_comparison <- print(compare(x = list(LOOIC_1,LOOIC_2,LOOIC_3,LOOIC_4,LOOIC_5,LOOIC_6,LOOIC_7)))
LOOIC_comparison


#plotting
{
# traceplot
traceplot(output, "pSW")
traceplot(output, "pSL")

# plot posteriors 
stan_plot(output, "pSW", show_density=T)
stan_plot(output, "pSL", show_density=T)

parameters <- rstan::extract(output)

}

#posterior predictive check
{parameters$Choice_pred[parameters$Choice_pred == -1] = NA

y_pred <- matrix(data = NA, nrow = 108, ncol = 26)
for(i in 1:26){
  for(t in 1:108){
    y_pred[t,i] <- mean(parameters$Choice_pred[,i,Advisor[t,i], ceiling(t/A)])
  }
}

y_pred

plot(Choice_raw[,6], type="l", xlab="Trial", ylab="Choice (0 or 1)", yaxt="n")
lines(y_pred[,6], col="red", lty=2)
axis(side=2, at = c(0,1) )
legend("bottomleft", legend=c("True", "PPC"), col=c("black", "red"), lty=1:2)
}

save(output1, output2, output3, output4, output5, output6, output7,
     LOOIC_1, LOOIC_2, LOOIC_3, LOOIC_4, LOOIC_5, LOOIC_6, LOOIC_7, LOOIC_comparison,
     file = "~/Yoonseo Zoh/2018-1/Computational Modeling/Final Project/Codes")
