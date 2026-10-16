# Explanations

Mode: `hetero`. Features are anonymised, so names are column names, not business meaning. Explanations describe the model, not causes.

- Stability (mean top-10 overlap across 3 runs): 0.42
- Deletion fidelity (hetero importances, 200 transactions): dropping the top-10 features changes the logit by 0.310 on average, against 0.091 for random features
- PyG fidelity metrics were not available for this model (see the warnings of the run)

## Transaction row 542000 (TransactionID 3529000)
- `V288`: 0.707
- `V90`: 0.704
- `freq_M9`: 0.703
- `D12`: 0.703
- `C7`: 0.701
- `V49`: 0.697
- `V304`: 0.694
- `dow_cos`: 0.693
- `V20`: 0.691
- `hour_cos`: 0.691
- edge `rev_owns`: 244625 -> 9633_-1 (0.849)
- edge `uses_device`: 379109 -> SM-G920P Build/NRD90M (0.821)
- edge `rev_owns`: 149065 -> 9633_-1 (0.810)
- edge `uses_device`: 541431 -> SM-G920P Build/NRD90M (0.809)
- edge `rev_owns`: 369087 -> 9633_-1 (0.808)

## Transaction row 542001 (TransactionID 3529001)
- `D12`: 0.730
- `D5`: 0.720
- `dow_cos`: 0.718
- `V75`: 0.712
- `V90`: 0.709
- `freq_M9`: 0.707
- `V12`: 0.705
- `V70`: 0.699
- `V49`: 0.696
- `freq_M8`: 0.695
- edge `rev_owns`: 216356 -> 12184_-1 (0.851)
- edge `uses_device`: 391700 -> SM-G920P Build/NRD90M (0.822)
- edge `rev_owns`: 527612 -> 12184_-1 (0.811)
- edge `uses_device`: 379152 -> SM-G920P Build/NRD90M (0.809)
- edge `rev_owns`: 48425 -> 12184_-1 (0.808)

## Transaction row 571931 (TransactionID 3558931)
- `V130`: 0.726
- `V49`: 0.725
- `V101`: 0.716
- `V304`: 0.693
- `V87`: 0.687
- `C10`: 0.685
- `V90`: 0.683
- `D12`: 0.679
- `V302`: 0.678
- `V38`: 0.673
- edge `rev_owns`: 49 -> 12730_-1 (0.850)
- edge `uses_device`: 91881 -> MotoG3 Build/MPIS24.65-33.1-2-16 (0.820)
- edge `rev_owns`: 376721 -> 12730_-1 (0.811)
- edge `uses_device`: 176034 -> MotoG3 Build/MPIS24.65-33.1-2-16 (0.809)
- edge `rev_owns`: 487971 -> 12730_-1 (0.809)

## Transaction row 583805 (TransactionID 3570805)
- `C7`: 0.712
- `V101`: 0.705
- `V90`: 0.701
- `V288`: 0.694
- `V85`: 0.681
- `V78`: 0.676
- `V105`: 0.671
- `D5`: 0.667
- `freq_M8`: 0.665
- `V289`: 0.654
- edge `rev_owns`: 214510 -> 5943_-1 (0.848)
- edge `uses_device`: 554747 -> Windows (0.822)
- edge `rev_owns`: 51343 -> 5943_-1 (0.810)
- edge `uses_device`: 93167 -> Windows (0.810)
- edge `rev_owns`: 64760 -> 5943_-1 (0.809)

## Transaction row 553431 (TransactionID 3540431)
- `hour_sin`: 0.738
- `V101`: 0.738
- `D5`: 0.734
- `V288`: 0.724
- `C7`: 0.694
- `V20`: 0.691
- `V40`: 0.690
- `V30`: 0.681
- `C5`: 0.679
- `V304`: 0.677
- edge `rev_owns`: 66108 -> 14076_-1 (0.851)
- edge `uses_device`: 390225 -> SM-G920P Build/NRD90M (0.822)
- edge `rev_owns`: 537727 -> 14076_-1 (0.811)
- edge `uses_device`: 541434 -> SM-G920P Build/NRD90M (0.810)
- edge `rev_owns`: 358440 -> 14076_-1 (0.808)

## Transaction row 565918 (TransactionID 3552918)
- `V101`: 0.722
- `V69`: 0.714
- `V90`: 0.702
- `C7`: 0.702
- `freq_M8`: 0.701
- `V75`: 0.692
- `V288`: 0.688
- `D7`: 0.688
- `V20`: 0.686
- `V29`: 0.683
- edge `rev_owns`: 214970 -> 1722_-1 (0.850)
- edge `uses_device`: 151898 -> Windows (0.821)
- edge `rev_owns`: 251855 -> 1722_-1 (0.811)
- edge `uses_device`: 26165 -> Windows (0.809)
- edge `rev_owns`: 301587 -> 1722_-1 (0.809)

## Transaction row 538595 (TransactionID 3525595)
- `freq_M9`: 0.745
- `V105`: 0.745
- `V36`: 0.733
- `na_D2`: 0.731
- `freq_M6`: 0.725
- `V49`: 0.723
- `V30`: 0.718
- `C14`: 0.718
- `V76`: 0.716
- `freq_P_emaildomain`: 0.715
- edge `rev_owns`: 512401 -> 10568_-1 (0.850)
- edge `uses_device`: 478140 -> LG-M700 Build/NMF26X (0.822)
- edge `rev_owns`: 359364 -> 10568_-1 (0.811)
- edge `uses_device`: 408726 -> LG-M700 Build/NMF26X (0.810)
- edge `rev_owns`: 122217 -> 10568_-1 (0.809)

## Transaction row 538594 (TransactionID 3525594)
- `freq_M9`: 0.753
- `V36`: 0.736
- `V105`: 0.726
- `hour_cos`: 0.724
- `freq_M6`: 0.720
- `D7`: 0.719
- `V76`: 0.709
- `freq_P_emaildomain`: 0.708
- `V30`: 0.707
- `na_D2`: 0.701
- edge `rev_owns`: 529627 -> 10568_-1 (0.847)
- edge `uses_device`: 527163 -> LG-M700 Build/NMF26X (0.822)
- edge `rev_owns`: 401018 -> 10568_-1 (0.811)
- edge `uses_device`: 448385 -> LG-M700 Build/NMF26X (0.810)
- edge `rev_owns`: 125937 -> 10568_-1 (0.808)

## Transaction row 538592 (TransactionID 3525592)
- `V288`: 0.715
- `freq_M9`: 0.713
- `V76`: 0.709
- `D9`: 0.697
- `freq_M6`: 0.693
- `V99`: 0.691
- `V64`: 0.691
- `V36`: 0.689
- `V30`: 0.689
- `V127`: 0.688
- edge `rev_owns`: 9906 -> 10568_-1 (0.848)
- edge `uses_device`: 106931 -> LG-M700 Build/NMF26X (0.822)
- edge `rev_owns`: 435674 -> 10568_-1 (0.811)
- edge `uses_device`: 487289 -> LG-M700 Build/NMF26X (0.810)
- edge `rev_owns`: 129595 -> 10568_-1 (0.808)

## Transaction row 530150 (TransactionID 3517150)
- `V38`: 0.690
- `freq_M2`: 0.688
- `V69`: 0.687
- `V288`: 0.681
- `V61`: 0.670
- `V126`: 0.670
- `C2`: 0.669
- `V315`: 0.668
- `C6`: 0.668
- `dow_cos`: 0.664
- edge `rev_owns`: 423142 -> 8389_-1 (0.886)
- edge `uses_email`: 441463 -> gmail.com (0.816)
- edge `uses_email`: 512933 -> gmail.com (0.806)
- edge `uses_email`: 32472 -> gmail.com (0.804)
- edge `uses_email`: 106414 -> gmail.com (0.804)

## User 4663_330 (account-risk head)
- edge `rev_owns`: 63656 -> 4663_330 (0.875)
- edge `rev_owns`: 123052 -> 4663_330 (0.858)
- edge `owns`: 4663_330 -> 193599 (0.835)
- edge `owns`: 4663_330 -> 97351 (0.821)
- edge `owns`: 4663_330 -> 323921 (0.820)

## User 8431_251 (account-risk head)
- edge `rev_owns`: 274919 -> 8431_251 (0.876)
- edge `rev_owns`: 160 -> 8431_251 (0.857)
- edge `owns`: 8431_251 -> 166536 (0.836)
- edge `rev_owns`: 334641 -> 8431_251 (0.823)
- edge `owns`: 8431_251 -> 12181 (0.821)

## User 13276_126 (account-risk head)
- edge `rev_owns`: 217452 -> 13276_126 (0.879)
- edge `rev_owns`: 410225 -> 13276_126 (0.853)
- edge `owns`: 13276_126 -> 321573 (0.836)
- edge `owns`: 13276_126 -> 83310 (0.821)
- edge `rev_owns`: 350629 -> 13276_126 (0.820)
