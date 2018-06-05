data {
int<lower=1> N;                                        // Number of subjects
int<lower=1> T;                                        // Number of trials across subjects
int<lower=1> A;                                        // number of nAdvisor
int<lower=-1, upper=1> Choice[N,A,T/A];                            
int<lower=0,upper=1> AdvisorCorrect[N,A,T/A];   // Outcome for each subject-block-trial
}

transformed data {
  vector[2] initV;
  initV = rep_vector(0.0, 2);
}

parameters {
  vector [2] mu_p;
  vector<lower=0>[2] sigma;
  vector [N] Eta_p;
  vector [N] Beta_p;
}

transformed parameters {
  vector<lower=0, upper=1> [N] Eta;
  vector<lower=0> [N] Beta;
  
  for (i in 1:N) {
    Eta[i] = Phi_approx(mu_p[1] + sigma[1]*Eta_p[i]);
    Beta[i] = exp(mu_p[2] + sigma[2]*Beta_p[i]);
  }
}

model {
  //individual parameters
  mu_p ~ normal(0,1);
  sigma ~ cauchy(0,5);
  Eta_p ~ normal(0,1);
  Beta_p ~ normal(0,1);
  
  for (i in 1:N) {
    for(a in 1:A) {
      
      vector[2] ev;
      real PE;
      
      ev = initV;
      
      for (t in 1:T/A) {
        int samesame;
        int result;
        
        if (Choice[i, a, t] == -1) continue;
        // Assign 0.5 to first trial
        if (t == 1) continue;
        else{
          if(Choice[i, a, t] == Choice[i, a, t-1]) samesame = 2;
          else samesame = 1;
          
          if(AdvisorCorrect[i, a, t] == Choice[i, a, t]) result = 1;
          else result = -1;
          
          samesame ~ categorical_logit(Beta[i]*ev);
          PE= result - ev[samesame];
          ev[samesame] = ev[samesame] + Eta[i]*PE;
        }
      }
    }
  }
}
generated quantities{
  real<lower=0, upper=1> mu_Eta;
  real<lower=0> mu_Beta;
  
  real Choice_pred[N,A,T/A];
  real log_lik[N];
  
  mu_Eta = Phi_approx(mu_p[1]);
  mu_Beta = exp(mu_p[2]);
  
  for (i in 1:N) {
    log_lik[i] = 0;
    for (a in 1:A) {
      
      vector[2] ev;
      real PE;
      
      ev = initV;
      
      for (t in 1:T/A) {
        int samesame;
        int result;
        
        if (Choice[i, a, t] == -1) {
          Choice_pred[i, a, t] = -1;
        }
        // Assign 0.5 to first trial
        else if (t == 1){
          Choice_pred[i, a, t] = categorical_rng((softmax(initV*Beta[i])))-1;
          if(Choice[i,a,t] == -1) continue;
        }
        else{
          if(Choice[i, a, t] == Choice[i, a, t-1]) samesame = 2;
          else samesame = 1;
          
          if(AdvisorCorrect[i, a, t] == Choice[i, a, t]) result = 1;
          else result = -1;
          
          if(Choice[i, a, t-1] == 1){
            Choice_pred[i, a, t] = categorical_rng((softmax(ev*Beta[i])))-1;
            if(Choice[i,a,t] == -1) continue;
            log_lik[i] = log_lik[i] + categorical_logit_lpmf(samesame | ev*Beta[i]);             
          }
          else{
            Choice_pred[i, a, t] = 2-categorical_rng((softmax(ev*Beta[i])));
            if(Choice[i,a,t] == -1) continue;
            log_lik[i] = log_lik[i] + categorical_logit_lpmf(samesame | ev*Beta[i]);             
          }
          PE = result - ev[samesame];
          ev[samesame] = ev[samesame] + Eta[i]*PE;
        }
      }
    }
  }
}
