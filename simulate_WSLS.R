setwd("~/Journey-of-Hyeonjin-Yoonseo/")

rm(list=ls())

# Simulation parameters
seed <- 0777    # do not change the seed number!
num_subjs  <- 26  # number of subjects
num_trials <- 36 # number of trials per subject
num_advisors <- 3
advisor_1_correct <- 0.25  # reward probability in option 1
advisor_2_correct <- 0.50  # reward probability in option 2
advisor_3_correct <- 0.75

# Set seed
set.seed(seed)   # always set a seed number for this homework!

# Generated True parameters 
{
simul_pars <- data.frame(pSW= rnorm(num_subjs, 0.85, 0.15),
                         pSL = rnorm(num_subjs, 0.40, 0.15),
                         subjID  = 1:num_subjs)

simul_pars$pSW[simul_pars$pSW < 0 | simul_pars$pSW > 1] = 0.85
simul_pars$pSL[simul_pars$pSL < 0 | simul_pars$pSL > 1] = 0.15
}

# True parameters with real data
{
parameters <- rstan::extract(output1)

pSWmean <- vector()
pSLmean <- vector()

for(i in 1:26){pSWmean[i] <- mean(parameters$pSW[,i])}
for(i in 1:26){pSLmean[i] <- mean(parameters$pSL[,i])}

simul_pars <- data.frame(pSWmean,
                         pSLmean,
                         subjID = 1:26)
}

{
  parameters <- rstan::extract(output1_h)
  
  pSWmean <- vector()
  pSLmean <- vector()
  
  for(i in 1:26){pSWmean[i] <- mean(parameters$pSW[,i])}
  for(i in 1:26){pSLmean[i] <- mean(parameters$pSL[,i])}
  
  simul_pars <- data.frame(pSWmean,
                           pSLmean,
                           subjID = 1:26)
}


Choice <- array(-1, c(num_subjs, num_advisors, num_trials))
AdvisorCorrect <- array(0, c(num_subjs, num_advisors, num_trials))

for (i in 1:num_subjs) {
  # Individual-level (i.e. per subject) parameter values
  pSW <- simul_pars$pSW[i]
  pSL <- simul_pars$pSL[i]
  
  AdvisorCorrect[i, 1, ] = rbinom(size = 1, n = 36, prob = advisor_1_correct)
  AdvisorCorrect[i, 2, ] = rbinom(size = 1, n = 36, prob = advisor_2_correct)
  AdvisorCorrect[i, 3, ] = rbinom(size = 1, n = 36, prob = advisor_2_correct)
  
  for (a in 1:num_advisors) {
    for (t in 1:num_trials) {
      if (t == 1) {
        Choice[i, a, t] = rbinom(size = 1, n = 1, prob = 0.5)
      }
      else {
        LastAdvisorCorrect = AdvisorCorrect[i, a, t-1]
        LastChoice = Choice[i, a, t-1]
        
        if (LastAdvisorCorrect == LastChoice) {
          if ((rbinom(size = 1, n = 1, prob = pSW)) == 1) {
            Choice[i, a, t] = LastChoice
            }
          else {
            if (Choice[i, a, t - 1] == 0) {
              Choice[i, a, t] = 1
            }
            else {
              Choice[i, a, t] = 0
            }
          }
        }
        else{
          if ((rbinom(size = 1, n = 1, prob = (1 - pSL)) == 1)) {
            Choice[i, a, t] = LastChoice
          }
          else{
            if (Choice[i, a, t-1] == 0) {
              Choice[i, a, t] = 1
            }
            else{
              Choice[i, a, t] = 0
            }
          }
        }
      }
    }
  }
}


  
dataList <- list(
  N = num_subjs,
  T = num_trials*num_advisors,
  A = num_advisors,
  Choice = Choice,
  AdvisorCorrect = AdvisorCorrect
)

output = stan("WSLS_classic.stan", data = dataList, pars=c("pSW", "pSL"),
              iter = 4000, warmup=2000, chains=4, cores=4)


output_h = stan("WSLS_classic_hierarchy.stan", data = dataList, pars=c("pSW", "pSL","mu_pSW", "mu_pSL"),
              iter = 4000, warmup=2000, chains=4, cores=4)

theme_set(theme_classic())

print(output)

parameters <- rstan::extract(output)

mean(parameters$pSW)
mean(parameters$pSL)

pSW_mean = apply(parameters$pSW, 2, mean)
pSW_sd = apply(parameters$pSW, 2, sd)
pSL_mean = apply(parameters$pSL, 2, mean)
pSL_sd = apply(parameters$pSL, 2, sd)

stan_plot(output, "pSW", show_density=T)
stan_plot(output, "pSL", show_density=T)

pSW_comparison <- data.frame(true = simul_pars$pSW, posterior = pSW_mean, posterior_sd = pSW_sd)
pSL_comparison <- data.frame(true = simul_pars$pSL, posterior = pSL_mean, posterior_sd = pSL_sd)


pSW <- ggplot(pSW_comparison, aes(x=true,y=posterior)) + geom_point(colour = "blue", size = 2)+
  geom_errorbar(aes(ymax=posterior+posterior_sd,ymin=posterior-posterior_sd,width=0)) +
  geom_abline(intercept=0, slope=1, color = "gray", linetype = "dashed", size = 0.5) +
  ggtitle("pSW")


pSL <- ggplot(pSL_comparison, aes(x=true,y=posterior)) + geom_point(colour = "blue", size = 2)+
  geom_errorbar(aes(ymax=posterior+posterior_sd,ymin=posterior-posterior_sd,width=0)) +
  geom_abline(intercept=0, slope=1, color = "gray", linetype = "dashed", size = 0.5) +
  ggtitle("pSL")



hierarchical_pSW <- ggplot(pSW_comparison, aes(x=true,y=posterior)) + geom_point(colour = "blue", size = 2)+
  geom_errorbar(aes(ymax=posterior+posterior_sd,ymin=posterior-posterior_sd,width=0)) +
  geom_abline(intercept=0, slope=1, color = "gray", linetype = "dashed", size = 0.5) +
  geom_hline(yintercept=mean(parameters$mu_pSW), slope =0, color = "red", linetype = "dashed", size=0.5)+
  ggtitle("hierarchical_pSW")

hierarchical_pSL <- ggplot(pSL_comparison, aes(x=true,y=posterior)) + geom_point(colour = "blue", size = 2)+
  geom_errorbar(aes(ymax=posterior+posterior_sd,ymin=posterior-posterior_sd,width=0)) +
  geom_abline(intercept=0, slope=1, color = "gray", linetype = "dashed", size = 0.5) +
  geom_hline(yintercept=mean(parameters$mu_pSL), slope =0, color = "red", linetype = "dashed", size=0.5)+
  ggtitle("hierarchical_pSL")



 
pSW+geom_smooth(method=lm, se=FALSE)
pSL+geom_smooth(method=lm, se=FALSE)
hierarchical_pSL+geom_smooth(method=lm, se=FALSE)
hierarchical_pSW+geom_smooth(method=lm, se=FALSE)

multiplot(hierarchical_pSW+geom_smooth(method=lm, se=FALSE),
          pSW+geom_smooth(method=lm, se=FALSE),
          hierarchical_pSL+geom_smooth(method=lm, se=FALSE),
          pSL+geom_smooth(method=lm, se=FALSE), cols= 2)

#multiplot function
{multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}}
