data {
  int<lower=1> N; //26
  int<lower=1> T; //36
  int<lower=1> A; //3
  int<lower=-1,upper=1> Choice[N,A,T/A];     
  int<lower=-1,upper=1> Outcome[N,A,T/A];   
}

transformed data {
  real initV = 0;  // initial values for EV
}

parameters {
  // Subject-level raw parameters (for Matt trick)
  vector<lower=0,upper=1>[N] Eta;
  vector[N] Beta_pr;
}

transformed parameters {
  vector<lower=0>[N] Beta;
  Beta = exp(Beta_pr);
}

model {
  // individual parameters
  Eta ~ normal(0,1);
  Beta_pr ~ normal(0,1);
  
  
  //
  for(i in 1:N){
      
    for(a in 1:A){ //n_advisor=3
      
      real V; // expected value
      real PE; // prediction error

      V = initV;
      
      for (t in 1:T/A) { //T=36
      
        if (Choice[i,a,t] == -1)
          continue;
        Choice[i,a,t] ~ bernoulli_logit(Beta[i] * V);
        PE = Outcome[i,a,t] - V;
        V = V + Eta[i] * PE;
      }
    }
  }
}

generated quantities {
  real Choice_pred[N,A,T/A];
  real log_lik[N];
  

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
