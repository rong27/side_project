function para_weight_pre_anfis_ans = para_weight_pre_anfis_model(paras_weight, fis)
    
    para_weight_pre_anfis = readfis(fis);
    para_weight_pre_anfis_ans = evalfis(para_weight_pre_anfis, paras_weight); 

end