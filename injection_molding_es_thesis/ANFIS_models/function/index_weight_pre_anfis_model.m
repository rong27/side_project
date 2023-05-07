function index_weight_pre_anfis_ans = index_weight_pre_anfis_model(index_weight, fis)
    
    index_weight_pre_anfis = readfis(fis);
    index_weight_pre_anfis_ans = evalfis(index_weight_pre_anfis, index_weight); 

end