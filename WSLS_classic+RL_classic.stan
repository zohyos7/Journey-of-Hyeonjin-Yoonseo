data {
  int<lower=1> N;                                        // Number of subjects
  int<lower=1> T;                                        // Number of trials across subjects
  int<lower=1> A;                                        // number of nAdvisor
  int<lower=-1, upper=1> Choice[N,A,T/A];                            
  int<lower=0,upper=1> AdvisorCorrect[N,A,T/A];
  int<lower=-1,upper=1> Outcome[N,A,T/A]; 
}

transformed data {
  real initV = 0;
}

parameters {
  vector [N] Eta_p;
  vector [N] Beta_p;
  vector [N] pSW_p;
  vector [N] pSL_p;
  vector [N] K_p;
}

transformed parameters {
  vector<lower=0, upper=1> [N] Eta;
  vector<lower=0> [N] Beta;
  vector<lower=0, upper=1> [N] pSW;
  vector<lower=0, upper=1> [N] pSL;
  vector<lower=0, upper=1> [N] K;
  
  for (i in 1:N){
    Eta[i] = Phi_approx(Eta_p[i]);
    Beta[i] = exp(Beta_p[i]);
    pSW[i] = Phi_approx(pSW_p[i]);
    pSL[i] = Phi_approx(pSL_p[i]);
    K[i] = Phi_approx(K_p[i]);  
  }
}

model {
  
  real pRL;
  real V; // expected value
  real PE; // prediction error
  //individual parameters
  
  pSW_p ~ normal(0, 1);
  pSL_p ~ normal(0, 1);
  Eta_p ~ normal(0, 1);
  Beta_p ~ normal(0, 1);
  K_p ~ normal(0, 1);


  for (i in 1:N) {
    for(a in 1:A) {
        V = initV;

      for (t in 1:T/A) {
        //RL
        if (Choice[i,a,t] == -1) continue;
        pRL = 1 / (1 + exp(-Beta[i] * V));
        
        //WSLS
        // Assign 0.5 to first trial
        if (t == 1) {
          Choice[i, a, t] ~ bernoulli(0.5*K[i] + pRL*(1-K[i]));
        }
        else if (Choice[i, a, t-1] == -1) continue;
        else {
          // Check if participant won last round
          if(AdvisorCorrect[i, a, t-1] == Choice[i, a, t-1]) {
            
            // Win Stay
            if(Choice[i, a, t-1] == 1){
              Choice[i, a, t] ~ bernoulli(pSW[i]*K[i] + pRL*(1-K[i]));
            }
            else{
              Choice[i, a, t] ~ bernoulli((1-pSW[i])*K[i] + pRL*(1-K[i]));  
            }
          }
          else {
            // Lose Stay
            if(Choice[i, a, t-1] == 1){
              Choice[i, a, t] ~ bernoulli((1-pSL[i])*K[i] + pRL*(1-K[i]));
            }
            else{
              Choice[i, a, t] ~ bernoulli(pSL[i]*K[i] + pRL*(1-K[i]));
            }
          }
        }
        
        PE = Outcome[i,a,t] - V;
        V = V + Eta[i] * PE;
        
      }
    }
  }  

}
generated quantities{
  real Choice_pred[N,A,T/A];    
  real log_lik[N];
  real pRL;
  real V; // expected value
  real PE; // prediction error

  
   {for (i in 1:N) {
      log_lik[i] = 0;
      for(a in 1:A) {
      V = initV;
      for (t in 1:T/A) {
        //RL
        pRL = 1 / (1 + exp(-Beta[i] * V));
        
        //WSLS
        // Assign 0.5 to first trial
        if (t == 1) {
          Choice_pred[i, a, t] = bernoulli_rng(0.5*K[i] + pRL*(1-K[i]));
          if(Choice[i,a,t] == -1) continue;
          log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t]|0.5*K[i] + pRL*(1-K[i]));
        }
        else if (Choice[i, a, t-1] == -1){
          Choice_pred[i, a, t] = -1;
          }
        else {
          // Check if participant won last round
          if(AdvisorCorrect[i, a, t-1] == Choice[i, a, t-1]) {
            // Win Stay
            if(Choice[i, a, t-1] == 1){
              Choice_pred[i, a, t] = bernoulli_rng(pSW[i]*K[i] + pRL*(1-K[i]));
              if(Choice[i, a, t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t]|pSW[i]*K[i] + pRL*(1-K[i]));
            }
            else{
              Choice_pred[i, a, t] = bernoulli_rng((1-pSW[i])*K[i] + pRL*(1-K[i]));
              if(Choice[i,a,t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t]|(1-pSW[i])*K[i] + pRL*(1-K[i]));     
            }
          }
          else {
            // Lose Stay
            if(Choice[i, a, t-1] == 1){
              Choice_pred[i, a, t] = bernoulli_rng((1-pSL[i])*K[i] + pRL*(1-K[i]));
              if(Choice[i, a, t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t]|(1-pSL[i])*K[i] + pRL*(1-K[i]));
            }
            else{
              Choice_pred[i, a, t] = bernoulli_rng(pSL[i]*K[i] + pRL*(1-K[i]));
              if(Choice[i, a, t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t]|pSL[i]*K[i] + pRL*(1-K[i]));
            }
          }
        }
        
        PE = Outcome[i,a,t] - V;
        V = V + Eta[i] * PE;
        
       }
     }
   }  
 }
}
