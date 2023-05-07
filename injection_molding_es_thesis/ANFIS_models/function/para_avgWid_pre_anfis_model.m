function para_avgWid_pre_anfis_ans = para_avgWid_pre_anfis_model(paras_avgWid, fis)
    
    para_avgWid_pre_anfis = readfis(fis);
    para_avgWid_pre_anfis_ans = evalfis(para_avgWid_pre_anfis, paras_avgWid); 

end