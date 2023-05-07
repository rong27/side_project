function index_avgLen_pre_anfis_ans = index_avgLen_pre_anfis_model(index_avgLen, fis)
    
    index_avgLen_pre_anfis = readfis(fis);
    index_avgLen_pre_anfis_ans = evalfis(index_avgLen_pre_anfis, index_avgLen); 

end