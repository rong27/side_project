function para_avgLen_pre_anfis_ans = para_avgLen_pre_anfis_model(paras_avgLen, fis)
    
    para_avgLen_pre_anfis = readfis(fis);
    para_avgLen_pre_anfis_ans = evalfis(para_avgLen_pre_anfis, paras_avgLen); 

end