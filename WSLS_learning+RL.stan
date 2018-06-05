data {
  int<lower=1> N;                                        // Number of subjects
  int<lower=1> T;                                        // Number of trials across subjects
  int<lower=1> A;                                        // number of nAdvisor
  int<lower=-1, upper=1> Choice[N,A,T/A];                            
  int<lower=0,upper=1> AdvisorCorrect[N,A,T/A];   // Outcome for each subject-block-trial
  int<lower=-1, upper=1> Outcome[N,A,T/A];
}

transformed data {
  real initV = 0;
}

parameters {
  vector [N] Eta_p;
  vector [N] Beta_p;
  vector [N] theta_pSW_p;
  vector [N] theta_pSL_p;
  vector [N] init_pSW_p;
  vector [N] init_pSL_p;
  vector [N] K_p;
}

transformed parameters {
  vector<lower=0, upper=1> [N] Eta;
  vector<lower=0> [N] Beta;
  vector<lower=0, upper=1> [N] theta_pSW;
  vector<lower=0, upper=1> [N] theta_pSL;
  vector<lower=0, upper=1> [N] init_pSW;
  vector<lower=0, upper=1> [N] init_pSL;
  vector<lower=0, upper=1> [N] K;
  
  for (i in 1:N){
    Eta[i] = Phi_approx(Eta_p[i]);
    Beta[i] = exp(Beta_p[i]);
    theta_pSW[i] = Phi_approx(theta_pSW_p[i]);
    theta_pSL[i] = Phi_approx(theta_pSL_p[i]);
    init_pSW[i] = Phi_approx(init_pSW_p[i]);
    init_pSL[i] = Phi_approx(init_pSL_p[i]);
    K[i] = Phi_approx(K_p[i]);
  }
}

model {
  real pSW;
  real pSL;
  real pRL;
  real V; // expected value
  real PE; // prediction error
  
  Eta_p ~ normal(0, 1);
  Beta_p ~ normal(0, 1);
  theta_pSW_p ~ normal(0, 1);
  theta_pSL_p ~ normal(0, 1);
  init_pSW_p ~ normal(0, 1);
  init_pSL_p ~ normal(0, 1);
  K_p ~ normal(0,1);
  
  for (i in 1:N) {
    for(a in 1:A) {
      
      V = initV;
      pSW = init_pSW[i];
      pSL = init_pSL[i];
      
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
            
            pSW = pSW + theta_pSW[i]*(1-pSW);
            pSL = (1-theta_pSL[i])*pSL;

            // Win Stay
            if(Choice[i, a, t-1] == 1){
              Choice[i, a, t] ~ bernoulli(pSW*K[i] + pRL*(1-K[i]));
            }
            else{
              Choice[i, a, t] ~ bernoulli((1-pSW)*K[i] + pRL*(1-K[i]));  
            }
          }
          else {
            pSL = pSL + theta_pSL[i]*(1-pSL);
            pSW = (1-theta_pSW[i])*pSW;

            // Lose Stay
            if(Choice[i,a,t-1] == 1){
              Choice[i,a,t] ~ bernoulli((1-pSL)*K[i] + pRL*(1-K[i]));
            }
            else{
              Choice[i,a,t] ~ bernoulli(pSL*K[i] + pRL*(1-K[i]));
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
  real pSW;
  real pSL;
  real pRL;
  real V; // expected value
  real PE; // prediction error
  real Choice_pred[N,A,T/A];
  real log_lik[N];
  
  for (i in 1:N) {
    log_lik[i] = 0;
    
    for(a in 1:A) {
      
      V = initV;
      pSW = init_pSW[i];
      pSL = init_pSL[i];
      
      for (t in 1:T/A) {
        // Assign 0.5 to first trial
        pRL = 1 / (1 + exp(-Beta[i] * V));
        
        if (t == 1) {
          Choice_pred[i, a, t] = bernoulli_rng(0.5*K[i] + pRL*(1-K[i]));
          if(Choice[i, a, t] == -1) continue;
          log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | (0.5*K[i] + pRL*(1-K[i])));
        }
        else if (Choice[i, a, t-1] == -1) {
          Choice_pred[i, a, t] = -1;
        }
        else {
          // Check if participant won last round
          if(AdvisorCorrect[i, a, t-1] == Choice_pred[i, a, t-1]) {
            pSW = pSW + theta_pSW[i]*(1-pSW);
            pSL = (1-theta_pSL[i])*pSL;

            // Win Stay
            if(Choice[i, a, t-1] == 1){
              Choice_pred[i, a, t] = bernoulli_rng(pSW*K[i] + pRL*(1-K[i]));
              if(Choice[i, a, t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | (pSW*K[i] + pRL*(1-K[i])) );
            }
            else{
              Choice_pred[i, a, t] = bernoulli_rng((1-pSW)*K[i] + pRL*(1-K[i]));
              if(Choice[i, a, t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | ((1-pSW)*K[i] + pRL*(1-K[i])) );
            }
          }
          else{
            pSL = pSL + theta_pSL[i]*(1-pSL);
            pSW = (1-theta_pSW[i])*pSW;

            // Lose Stay
            if(Choice[i, a, t-1] == 1){
              Choice_pred[i, a, t] = bernoulli_rng((1-pSL)*K[i] + pRL*(1-K[i]));
              if(Choice[i, a, t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | ((1-pSL)*K[i] + pRL*(1-K[i])) );
            }
            else{
              Choice_pred[i, a, t] = bernoulli_rng(pSL*K[i] + pRL*(1-K[i]));
              if(Choice[i, a, t] == -1) continue;
              log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | (pSL*K[i] + pRL*(1-K[i])) );
            }  
          }
        }
        
        PE = Outcome[i,a,t] - V;
        V = V + Eta[i] * PE;
        
      }
    }
  }
}
