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
  vector[N] Eta_pr;
  vector[N] Beta_pr;
}

transformed parameters {
  vector<lower=0, upper=1> [N] Eta;
  vector<lower=0> [N] Beta;

  for (i in 1:N) {
    Eta[i] = Phi_approx(Eta_pr[i]);
    Beta[i] = exp(Beta_pr[i]);
  }
}

model {
  //individual parameters
  Eta_pr ~ normal(0,1);
  Beta_pr ~ normal(0,1);
  
  for (i in 1:N) {
    for(a in 1:A) {
      
      vector[2] w_ev;
      vector[2] l_ev;
      real PE;
      
      w_ev = initV;
      l_ev = initV;
      
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
           // Check if participant won last round
           if(AdvisorCorrect[i, a, t-1] == Choice[i, a, t-1]) {
            // Win
            // Win Stay
            samesame ~ categorical_logit(Beta[i]*w_ev);
            PE= result - w_ev[samesame];
            w_ev[samesame] = w_ev[samesame] + Eta[i]*PE;
            }
          else {
            // Lose
            // Lose Stay
            samesame ~ categorical_logit(Beta[i]*l_ev);
            PE = result - l_ev[samesame];
            l_ev[samesame] = l_ev[samesame] + Eta[i]*PE;
          }
        }
      }
    }
  }
}
generated quantities{
  real Choice_pred[N,A,T/A];
  real log_lik[N];
  
  for (i in 1:N) {
    log_lik[i] = 0;
    for (a in 1:A) {
      
      vector[2] w_ev;
      vector[2] l_ev;

      real PE;
      
      w_ev = initV;
      l_ev = initV;

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
           // Check if participant won last round
           if(AdvisorCorrect[i, a, t-1] == Choice[i, a, t-1]) {
            // Win
            // Win Stay
            if(Choice[i, a, t-1] == 1){
              Choice_pred[i, a, t] = categorical_rng((softmax(w_ev*Beta[i])))-1;
              if(Choice[i,a,t] == -1) continue;
              log_lik[i] = log_lik[i] + categorical_logit_lpmf(samesame | w_ev*Beta[i]);              
            }
            else{
              Choice_pred[i, a, t] = 2-categorical_rng((softmax(w_ev*Beta[i])));
              if(Choice[i,a,t] == -1) continue;
              log_lik[i] = log_lik[i] + categorical_logit_lpmf(samesame | w_ev*Beta[i]);                           
             }
            PE= result - w_ev[samesame];
            w_ev[samesame] = w_ev[samesame] + Eta[i]*PE;
            }
           else {
            // Lose
            // Lose Stay
            if(Choice[i, a, t-1] == 1){
              Choice_pred[i, a, t] = categorical_rng((softmax(l_ev*Beta[i])))-1;
              if(Choice[i,a,t] == -1) continue;
              log_lik[i] = log_lik[i] + categorical_logit_lpmf(samesame | l_ev*Beta[i]);             
             }
            else{
              Choice_pred[i, a, t] = 2-categorical_rng((softmax(l_ev*Beta[i])));
              if(Choice[i,a,t] == -1) continue;
              log_lik[i] = log_lik[i] + categorical_logit_lpmf(samesame | l_ev*Beta[i]);             
            }
            PE = result - l_ev[samesame];
            l_ev[samesame] = l_ev[samesame] + Eta[i]*PE;
          }
        }
      }
    }
  }
}

