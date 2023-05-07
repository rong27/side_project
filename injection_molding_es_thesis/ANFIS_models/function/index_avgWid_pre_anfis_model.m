function index_avgWid_pre_anfis_ans = index_avgWid_pre_anfis_model(index_avgWid, fis)
    
    index_avgWid_pre_anfis = readfis(fis);
    index_avgWid_pre_anfis_ans = evalfis(index_avgWid_pre_anfis, index_avgWid); 

end