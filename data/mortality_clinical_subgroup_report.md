# Mortality clinical subgroup report

**Test n:** 1250 | **AUROC:** 0.9154 | **AUPRC:** 0.8369

**Safety:** High-SCAI tertile: AUROC below overall is expected at ~74% death rate; AUPRC 0.910 supports ranking among high-risk patients. See mortality_presentation_signoff.md.
**High-SCAI review:** complete

## SCAI tertile (current_scai)

| Tertile | n | death_rate | AUROC | AUPRC | Δ AUROC |
|---------|--:|-----------:|------:|------:|--------:|
| (2.0, 4.0] | 377 | 73.2% | 0.756 | 0.910 | -0.160 |
| (-0.001, 2.0] | 873 | 9.5% | 0.828 | 0.371 | -0.088 |

## Primary diagnosis (top groups)

- **CLASSIFICATION_CD=PRINCIPAL** n=1250 AUROC=0.915 (Δ -0.000)
- **NOMENCLATURE_CD=I21.4** n=241 AUROC=0.927 (Δ +0.011)
- **NOMENCLATURE_CD=I50.21** n=200 AUROC=0.919 (Δ +0.004)
- **NOMENCLATURE_CD=I21.3** n=195 AUROC=0.891 (Δ -0.024)
- **NOMENCLATURE_CD=I50.23** n=151 AUROC=0.915 (Δ -0.001)
- **NOMENCLATURE_CD=Z95.1** n=85 AUROC=0.921 (Δ +0.006)
- **NOMENCLATURE_CD=I40.9** n=72 AUROC=0.975 (Δ +0.060)
- **NOMENCLATURE_CD=I35.0** n=70 AUROC=0.895 (Δ -0.020)
- **NOMENCLATURE_CD=I46.9** n=67 AUROC=0.929 (Δ +0.013)
- **NOMENCLATURE_CD=I42.0** n=65 AUROC=0.843 (Δ -0.073)
- **NOMENCLATURE_CD=I71.01** n=49 AUROC=0.849 (Δ -0.067)
- **NOMENCLATURE_CD=I49.01** n=33 AUROC=0.808 (Δ -0.108)
