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
  vector[2] mu_p;
  vector<lower=0>[2] sigma;
  
  vector[N] pSW_pr;
  vector[N] pSL_pr;
}

transformed parameters {
  vector<lower=0, upper=1> [N] pSW;
  vector<lower=0, upper=1> [N] pSL;
  
  for (i in 1:N) {
    pSW[i] = Phi_approx(mu_p[1] + sigma[1]*pSW_pr[i]);
    pSL[i] = Phi_approx(mu_p[2] + sigma[2]*pSL_pr[i]);
  }
  
}

model {
  //Hyperparameters
  mu_p ~ normal(0, 1);
  sigma ~ cauchy(0, 5);
  
  //individual parameters
  pSW_pr ~ normal(0, 1);
  pSL_pr ~ normal(0, 1);
  
  
  for (i in 1:N) {
    for(a in 1:A) {
      for (t in 1:T/A) {
        int samesame;
        if (Choice[i, a, t] == -1) continue;
        // Assign 0.5 to first trial
        if (t == 1) continue;
        else if (Choice[i, a, t-1] == -1) continue;
        else {
          if(Choice[i, a, t] == Choice[i, a, t-1]) samesame = 1;
          else samesame = 0;
          // Check if participant won last round
          if(AdvisorCorrect[i, a, t-1] == Choice[i, a, t-1]) {
            // Win
            // Win Stay
            samesame ~ bernoulli(pSW[i]);
          }
          else {
            // Lose
            // Lose Stay
            samesame ~ bernoulli(1 - pSL[i]);
          }
        }
      }
    }
  }
}
generated quantities{
  real Choice_pred[N,A,T/A];    
  real log_lik[N];
  
  {
   for (i in 1:N) {
     log_lik[i] = 0;
     for(a in 1:A) {
      for (t in 1:T/A) {
        if (Choice[i, a, t] == -1) {
         Choice_pred[i, a, t] = -1;
         }
        // Assign 0.5 to first trial
        if (t == 1) {
         Choice_pred[i, a, t] = bernoulli_rng(0.5);
         if (Choice[i, a, t] == -1) continue;         
         log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i, a, t] | 0.5) ;
         
          }
        else if (Choice[i, a, t-1] == -1) {
         Choice_pred[i, a, t] = -1;
        }
        else {
          if(AdvisorCorrect[i, a, t-1] == Choice[i, a, t-1]) {
            if (Choice[i, a, t-1] == 1){
             Choice_pred[i, a, t] = bernoulli_rng(pSW[i]);
             if (Choice[i, a, t] == -1) continue;
             log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i, a, t] | pSW[i]) ;
            }
            else {
             Choice_pred[i, a, t] = bernoulli_rng(1-pSW[i]);
             if (Choice[i, a, t] == -1) continue;
             log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i, a, t] | 1-pSW[i]) ;
            }
          }
          else {
            if (Choice[i, a, t-1] == 0){
             Choice_pred[i, a, t] = bernoulli_rng(pSL[i]);
             if (Choice[i, a, t] == -1) continue;
             log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i, a, t] | pSL[i]) ;
             
            }
            else {
             Choice_pred[i, a, t] = bernoulli_rng(1-pSL[i]);
             if (Choice[i, a, t] == -1) continue;
             log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i, a, t] | 1-pSL[i]) ;
             }
           }
         }
       }
     }
   }
 }
}

