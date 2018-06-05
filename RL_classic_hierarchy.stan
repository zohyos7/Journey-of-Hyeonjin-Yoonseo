// model with alpha and beta
data {
  int<lower=1> N; //26
  int<lower=1> T; //108
  int<lower=1> A; //3
  int<lower=-1,upper=1> Choice[N,A,T/A];     
  int<lower=-1,upper=1> Outcome[N,A,T/A];   
}

transformed data {
  real initV;  // initial values for EV
  initV = 0;
}

parameters {
  // Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters  
  vector[2] mu_p;  
  vector<lower=0>[2] sigma;
  // Subject-level raw parameters (for Matt trick)
  vector[N] Eta_pr;    // learning rate [0, 1]
  vector[N] Beta_pr;  // inverse temperature [0, 5]
}

transformed parameters {
  // subject-level parameters
  vector<lower=0,upper=1>[N] Eta;
  vector<lower=0>[N] Beta; //I'm not sure about 25.
  
  for (i in 1:N) {
    Eta[i] = Phi_approx( mu_p[1] + sigma[1] * Eta_pr[i] );
    Beta[i] = exp( mu_p[2] + sigma[2] * Beta_pr[i] );
  }
}

model {
  // Hyperparameters
  mu_p  ~ normal(0, 1); 
  sigma ~ cauchy(0, 5);
  
  // individual parameters
  Eta_pr ~ normal(0,1);
  Beta_pr ~ normal(0,1);
  
  //
  for(i in 1:N){
      
    for(a in 1:A){ 
      
      real V; // expected value
      real PE; // prediction error
      V = initV;
      
      for (t in 1:T/A) { //T=108
        if (Choice[i,a,t] == -1) continue;
          
        Choice[i,a,t] ~ bernoulli_logit(Beta[i] * V);
        PE = Outcome[i,a,t] - V;
        V = V + Eta[i] * PE;
      }
    }
  }
}
generated quantities {
  real<lower=0, upper=1> mu_Eta;
  real<lower=0> mu_Beta;
  
  real Choice_pred[N,A,T/A];
  real log_lik[N];
  
  mu_Eta = Phi_approx(mu_p[1]);
  mu_Beta = exp(mu_p[2]);

  for (i in 1:N) {
    log_lik[i] = 0;
    for(a in 1:A) {
      real V;
      real PE;
      V = initV;
      
      for (t in 1:T/A) {
        Choice_pred[i,a,t] = bernoulli_logit_rng(Beta[i] * V);
        if (Choice[i,a,t] == -1) {
        PE = Outcome[i,a,t] - V;
        V = V + Eta[i] * PE;
        }
        else{
        log_lik[i] = log_lik[i] + bernoulli_lpmf(Choice[i,a,t] | inv_logit(Beta[i]*V));
        PE = Outcome[i,a,t] - V;
        V = V + Eta[i] * PE;
        }
      }
    }
  }
}

