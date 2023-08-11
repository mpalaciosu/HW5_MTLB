function delta = getdelta(q, T, d1, t)

delta = exp(-q * (T - t)) * normcdf(d1);

return
