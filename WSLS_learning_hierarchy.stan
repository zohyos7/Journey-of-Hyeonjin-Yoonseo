data {
  int<lower=1> N;                                        // Number of subjects
  int<lower=1> T;                                        // Number of trials across subjects
  int<lower=1> A;                                        // number of nAdvisor
  int<lower=-1, upper=1> Choice[N,A,T/A];                            
  int<lower=0,upper=1> AdvisorCorrect[N,A,T/A];   // Outcome for each subject-block-trial
}

transformed data {
}

parameters {
  //hyper parameters
  vector[4] mu_p;
  vector<lower=0> [4] sigma;
  
  //individual parameters
  vector [N] theta_pSW_pr;
  vector [N] theta_pSL_pr;
  vector [N] init_pSW_pr;
  vector [N] init_pSL_pr;
}

transformed parameters {
  vector<lower=0, upper=1> [N] theta_pSW;
  vector<lower=0, upper=1> [N] theta_pSL;
  vector<lower=0, upper=1> [N] init_pSW;
  vector<lower=0, upper=1> [N] init_pSL;
  
  for(i in 1:N){
    theta_pSW[i] = Phi_approx(mu_p[1] + sigma[1]*theta_pSW_pr[i]);
    theta_pSL[i] = Phi_approx(mu_p[2] + sigma[2]*theta_pSL_pr[i]);
    init_pSW[i] = Phi_approx(mu_p[3] + sigma[3]*init_pSW_pr[i]);
    init_pSL[i] = Phi_approx(mu_p[4] + sigma[4]*init_pSL_pr[i]);
  }
  
}

model {  
  vector[T/A] pSW;
  vector[T/A] pSL;
  
  mu_p  ~ normal(0, 1); 
  sigma ~ cauchy(0, 5);
  
  theta_pSW_pr ~ normal(0, 1);
  theta_pSL_pr ~ normal(0, 1);
  init_pSW_pr ~ normal(0, 1);
  init_pSL_pr ~ normal(0, 1); // I changed it from Beta(1,1) to normal(0,1)
  
  for (i in 1:N) {
    for(a in 1:A) {
      for (t in 1:T/A) {
        int samesame;
        if (Choice[i, a, t] == -1) continue;
        // Assign 0.5 to first trial
        if (t == 1) {
          Choice[i, a, t] ~ bernoulli(0.5);
        }
        else if (Choice[i, a, t-1] == -1) continue;
        else if (t == 2) {
          if(Choice[i, a, t] == Choice[i, a, t-1]) samesame = 1;
          else samesame = 0;
          // Check if participant won last round
          if(AdvisorCorrect[i, a, t-1] == Choice[i, a, t-1]) {
            // Win
            // Win Stay
            samesame ~ bernoulli(init_pSW[i]);
            pSW[t] = init_pSW[i] + theta_pSW[i]*(1-init_pSW[i]);
            pSL[t] = (1-theta_pSL[i])*init_pSL[i];
          }
          else {
            // Lose
            // Lose Stay
            samesame ~ bernoulli(1 - init_pSL[i]);
            pSL[t] = init_pSL[i] + theta_pSL[i]*(1-init_pSL[i]);
            pSW[t] = (1-theta_pSW[i])*init_pSW[i];
          }
        }
        else {
          if(Choice[i, a, t] == Choice[i, a, t-1]) samesame = 1;
          else samesame = 0;
          // Check if participant won last round
          if(AdvisorCorrect[i, a, t-1] == Choice[i, a, t-1]) {
            // Win
            // Win Stay
            samesame ~ bernoulli(pSW[t-1]);
            pSW[t] = pSW[t-1] + theta_pSW[i]*(1-pSW[t-1]);
            pSL[t] = (1-theta_pSL[i])*pSL[t-1];
          }
          else {
            // Lose
            // Lose Stay
            samesame ~ bernoulli(1 - pSL[t-1]);
            pSL[t] = pSL[t-1] + theta_pSL[i]*(1-pSL[t-1]);
            pSW[t] = (1-theta_pSW[i])*pSW[t-1];
          }
        }
      }
    }
  }
}
generated quantities{
  real<lower=0, upper=1> mu_theta_pSW;
  real<lower=0, upper=1> mu_theta_pSL;
  real<lower=0, upper=1> mu_init_pSW;
  real<lower=0, upper=1> mu_init_pSL;
  
  vector[T/A] pSW;
  vector[T/A] pSL;
  real Choice_pred[N,A,T/A];
  real log_lik[N];
  
  mu_theta_pSW = Phi_approx(mu_p[1]);
  mu_theta_pSL = Phi_approx(mu_p[2]);
  mu_init_pSW = Phi_approx(mu_p[3]);
  mu_init_pSL = Phi_approx(mu_p[4]);
  
  for (i in 1:N) {
    log_lik[i] = 0;
    for (a in 1:A) {
      for (t in 1:T/A) {
        if (Choice[i, a, t] == -1) {
          Choice_pred[i, a, t] = -1;
        }
        // Assign 0.5 to first trial
        if (t == 1) {
          Choice_pred[i, a, t] = bernoulli_rng(0.5);
          if(Choice[i,a,t] == -1) continue;
          log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | 0.5);
        }
        else if (Choice[i, a, t-1] == -1) {
          Choice_pred[i, a, t] = -1;
        }
        else if (t == 2) {
          // Check if participant won last round
          if(AdvisorCorrect[i, a, t-1] == Choice[i, a, t-1]) {
            // Win
            // Win Stay
            pSW[t] = init_pSW[i] + theta_pSW[i]*(1-init_pSW[i]);
            pSL[t] = (1-theta_pSL[i])*init_pSL[i];
            
            if(Choice[i, a, t-1] == 1){
              Choice_pred[i, a, t] = bernoulli_rng(init_pSW[i]);
              if(Choice[i,a,t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | init_pSW[i]);
              
            }
            else{
              Choice_pred[i, a, t] = bernoulli_rng(1-init_pSW[i]);
              if(Choice[i,a,t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | 1-init_pSW[i]);
            }
          }
          else {
            // Lose
            // Lose Stay
            pSL[t] = init_pSL[i] + theta_pSL[i]*(1-init_pSL[i]);
            pSW[t] = (1-theta_pSW[i])*init_pSW[i];
            
            if(Choice[i, a, t-1] == 1){
              Choice_pred[i, a, t] = bernoulli_rng(1-init_pSL[i]);
              if(Choice[i,a,t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | 1-init_pSL[i]);
            }
            else{
              Choice_pred[i, a, t] = bernoulli_rng(init_pSL[i]);
              if(Choice[i,a,t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | init_pSL[i]);
            }
          }
        }
        else {
          // Check if participant won last round
          if(AdvisorCorrect[i, a, t-1] == Choice[i, a, t-1]) {
            // Win
            // Win Stay
            pSW[t] = pSW[t-1] + theta_pSW[i]*(1-pSW[t-1]);
            pSL[t] = (1-theta_pSL[i])*pSL[t-1];
            
            if(Choice[i, a, t-1] == 1){
              Choice_pred[i, a, t] = bernoulli_rng(pSW[t-1]);
              if(Choice[i,a,t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | pSW[t-1]);
              
            }
            else{
              Choice_pred[i, a, t] = bernoulli_rng(1-pSW[t-1]);
              if(Choice[i,a,t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | 1-pSW[t-1]);
            }
          }
          else{
            // Lose
            // Lose Stay
            pSL[t] = pSL[t-1] + theta_pSL[i]*(1-pSL[t-1]);
            pSW[t] = (1-theta_pSW[i])*pSW[t-1];
            
            if(Choice[i, a, t-1] == 1){
              Choice_pred[i, a, t] = bernoulli_rng(1-pSL[t-1]);
              if(Choice[i,a,t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | 1-pSL[t-1]);
            }
            else{
              Choice_pred[i, a, t] = bernoulli_rng(pSL[t-1]);
              if(Choice[i,a,t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | pSL[t-1]);
            }
          }
        }
      }
    }
  }
}
