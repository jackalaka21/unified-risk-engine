# Explanations

Mode: `hetero`. Features are anonymised, so names are column names, not business meaning. Explanations describe the model, not causes.

- Stability (mean top-10 overlap across 3 runs): 0.40
- Deletion fidelity (hetero importances, 200 transactions): dropping the top-10 features changes the logit by 0.340 on average, against 0.092 for random features
- PyG fidelity metrics were not available for this model (see the warnings of the run)

## Transaction row 553431 (TransactionID 3540431)
- `V288`: 0.744
- `V101`: 0.719
- `D5`: 0.688
- `C7`: 0.684
- `V105`: 0.664
- `V20`: 0.658
- `V40`: 0.658
- `V49`: 0.656
- `log_amt`: 0.655
- `V90`: 0.655
- edge `rev_owns`: 438267 -> 14076_-1 (0.849)
- edge `uses_device`: 64396 -> SM-G920P Build/NRD90M (0.821)
- edge `rev_owns`: 454472 -> 14076_-1 (0.810)
- edge `rev_owns`: 407595 -> 14076_-1 (0.810)
- edge `uses_device`: 112753 -> SM-G920P Build/NRD90M (0.809)

## Transaction row 535630 (TransactionID 3522630)
- `freq_M2`: 0.736
- `V44`: 0.710
- `V304`: 0.702
- `C2`: 0.698
- `V290`: 0.685
- `C8`: 0.682
- `hour_sin`: 0.681
- `V45`: 0.679
- `C10`: 0.676
- `C1`: 0.674
- edge `rev_owns`: 16349 -> 14076_-1 (0.865)
- edge `rev_owns`: 227443 -> 14076_-1 (0.814)
- edge `rev_owns`: 518047 -> 14076_-1 (0.813)
- edge `uses_email`: 383070 -> gmail.com (0.811)
- edge `uses_email`: 58214 -> gmail.com (0.804)

## Transaction row 542000 (TransactionID 3529000)
- `dow_cos`: 0.739
- `V53`: 0.716
- `C7`: 0.709
- `V288`: 0.706
- `V90`: 0.705
- `V49`: 0.694
- `V75`: 0.675
- `hour_cos`: 0.673
- `C2`: 0.671
- `freq_P_emaildomain`: 0.670
- edge `rev_owns`: 301446 -> 9633_-1 (0.850)
- edge `uses_device`: 214250 -> SM-G920P Build/NRD90M (0.822)
- edge `rev_owns`: 397273 -> 9633_-1 (0.811)
- edge `uses_device`: 541385 -> SM-G920P Build/NRD90M (0.810)
- edge `rev_owns`: 137783 -> 9633_-1 (0.808)

## Transaction row 542001 (TransactionID 3529001)
- `dow_cos`: 0.736
- `D5`: 0.726
- `V304`: 0.719
- `V288`: 0.717
- `freq_M8`: 0.701
- `C7`: 0.701
- `hour_cos`: 0.694
- `V49`: 0.688
- `D10`: 0.687
- `freq_P_emaildomain`: 0.686
- edge `rev_owns`: 216356 -> 12184_-1 (0.851)
- edge `uses_device`: 391700 -> SM-G920P Build/NRD90M (0.822)
- edge `rev_owns`: 527612 -> 12184_-1 (0.811)
- edge `uses_device`: 379152 -> SM-G920P Build/NRD90M (0.810)
- edge `rev_owns`: 48425 -> 12184_-1 (0.810)

## Transaction row 571931 (TransactionID 3558931)
- `V49`: 0.738
- `V90`: 0.713
- `V130`: 0.712
- `V101`: 0.705
- `V87`: 0.688
- `V45`: 0.683
- `C2`: 0.681
- `V287`: 0.678
- `D8`: 0.678
- `V105`: 0.677
- edge `rev_owns`: 49 -> 12730_-1 (0.850)
- edge `uses_device`: 91881 -> MotoG3 Build/MPIS24.65-33.1-2-16 (0.820)
- edge `rev_owns`: 376721 -> 12730_-1 (0.811)
- edge `rev_owns`: 487971 -> 12730_-1 (0.809)
- edge `uses_device`: 176034 -> MotoG3 Build/MPIS24.65-33.1-2-16 (0.809)

## Transaction row 565918 (TransactionID 3552918)
- `V288`: 0.703
- `V85`: 0.689
- `V99`: 0.686
- `hour_cos`: 0.668
- `C7`: 0.667
- `D15`: 0.666
- `V289`: 0.665
- `na_D11`: 0.664
- `freq_M8`: 0.660
- `V30`: 0.659
- edge `rev_owns`: 214970 -> 1722_-1 (0.850)
- edge `uses_device`: 151898 -> Windows (0.820)
- edge `rev_owns`: 251855 -> 1722_-1 (0.811)
- edge `rev_owns`: 301587 -> 1722_-1 (0.809)
- edge `uses_device`: 26165 -> Windows (0.809)

## Transaction row 538595 (TransactionID 3525595)
- `freq_M9`: 0.740
- `V49`: 0.732
- `V30`: 0.721
- `na_D2`: 0.714
- `freq_P_emaildomain`: 0.713
- `V69`: 0.710
- `hour_cos`: 0.709
- `D13`: 0.707
- `V76`: 0.706
- `hour_sin`: 0.700
- edge `rev_owns`: 512401 -> 10568_-1 (0.850)
- edge `uses_device`: 478140 -> LG-M700 Build/NMF26X (0.822)
- edge `rev_owns`: 359364 -> 10568_-1 (0.811)
- edge `uses_device`: 408726 -> LG-M700 Build/NMF26X (0.810)
- edge `rev_owns`: 122217 -> 10568_-1 (0.809)

## Transaction row 538594 (TransactionID 3525594)
- `V49`: 0.732
- `V30`: 0.711
- `V69`: 0.709
- `D13`: 0.700
- `hour_cos`: 0.698
- `V75`: 0.694
- `hour_sin`: 0.694
- `V44`: 0.694
- `freq_P_emaildomain`: 0.691
- `na_D2`: 0.691
- edge `rev_owns`: 529627 -> 10568_-1 (0.849)
- edge `uses_device`: 527163 -> LG-M700 Build/NMF26X (0.822)
- edge `rev_owns`: 401018 -> 10568_-1 (0.811)
- edge `uses_device`: 448385 -> LG-M700 Build/NMF26X (0.810)
- edge `rev_owns`: 125937 -> 10568_-1 (0.808)

## Transaction row 538592 (TransactionID 3525592)
- `V20`: 0.734
- `freq_M9`: 0.720
- `D5`: 0.710
- `V64`: 0.705
- `V49`: 0.702
- `V69`: 0.698
- `D7`: 0.692
- `freq_M6`: 0.692
- `V75`: 0.691
- `V105`: 0.687
- edge `rev_owns`: 9906 -> 10568_-1 (0.848)
- edge `uses_device`: 106931 -> LG-M700 Build/NMF26X (0.822)
- edge `rev_owns`: 435674 -> 10568_-1 (0.811)
- edge `uses_device`: 487289 -> LG-M700 Build/NMF26X (0.810)
- edge `rev_owns`: 129595 -> 10568_-1 (0.809)

## Transaction row 530150 (TransactionID 3517150)
- `V86`: 0.704
- `V69`: 0.701
- `V314`: 0.688
- `V288`: 0.684
- `V61`: 0.678
- `D12`: 0.677
- `C6`: 0.675
- `V38`: 0.670
- `V49`: 0.669
- `V60`: 0.669
- edge `rev_owns`: 423142 -> 8389_-1 (0.886)
- edge `uses_email`: 441463 -> gmail.com (0.816)
- edge `uses_email`: 512933 -> gmail.com (0.807)
- edge `uses_email`: 32472 -> gmail.com (0.804)
- edge `uses_email`: 106414 -> gmail.com (0.804)

## User 4663_330 (account-risk head)
- edge `rev_owns`: 63656 -> 4663_330 (0.876)
- edge `rev_owns`: 123052 -> 4663_330 (0.856)
- edge `owns`: 4663_330 -> 193599 (0.835)
- edge `owns`: 4663_330 -> 323921 (0.823)
- edge `owns`: 4663_330 -> 97351 (0.821)

## User 8431_251 (account-risk head)
- edge `rev_owns`: 274919 -> 8431_251 (0.876)
- edge `rev_owns`: 160 -> 8431_251 (0.856)
- edge `owns`: 8431_251 -> 166536 (0.836)
- edge `rev_owns`: 334641 -> 8431_251 (0.823)
- edge `owns`: 8431_251 -> 140730 (0.820)

## User 13276_126 (account-risk head)
- edge `rev_owns`: 217452 -> 13276_126 (0.877)
- edge `rev_owns`: 410225 -> 13276_126 (0.854)
- edge `owns`: 13276_126 -> 321573 (0.836)
- edge `owns`: 13276_126 -> 83310 (0.820)
- edge `owns`: 13276_126 -> 347519 (0.820)
