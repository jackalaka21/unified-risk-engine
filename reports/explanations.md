# Explanations

Mode: `hetero`. Features are anonymised, so names are column names, not business meaning. Explanations describe the model, not causes.

- Stability (mean top-10 overlap across 3 runs): 0.42
- Deletion fidelity (hetero importances, 200 transactions): dropping the top-10 features changes the logit by 0.243 on average, against 0.065 for random features
- PyG fidelity metrics were not available for this model (see the warnings of the run)

## Transaction row 551943 (TransactionID 3538943)
- `V288`: 0.736
- `V130`: 0.709
- `V44`: 0.689
- `V85`: 0.678
- `na_V62`: 0.676
- `V45`: 0.674
- `V49`: 0.671
- `C13`: 0.669
- `V310`: 0.668
- `V30`: 0.664
- edge `rev_owns`: 254098 -> 9026_-1 (0.849)
- edge `uses_device`: 401673 -> MotoG3 Build/MPIS24.65-33.1-2-16 (0.822)
- edge `uses_device`: 148098 -> MotoG3 Build/MPIS24.65-33.1-2-16 (0.810)
- edge `rev_owns`: 161366 -> 9026_-1 (0.810)
- edge `rev_owns`: 79560 -> 9026_-1 (0.809)

## Transaction row 583805 (TransactionID 3570805)
- `V288`: 0.714
- `na_V62`: 0.701
- `freq_M6`: 0.699
- `V302`: 0.693
- `hour_cos`: 0.685
- `D5`: 0.679
- `V90`: 0.672
- `C7`: 0.670
- `V30`: 0.670
- `V85`: 0.663
- edge `rev_owns`: 214510 -> 5943_-1 (0.848)
- edge `uses_device`: 554747 -> Windows (0.822)
- edge `rev_owns`: 51343 -> 5943_-1 (0.810)
- edge `uses_device`: 93167 -> Windows (0.809)
- edge `rev_owns`: 64760 -> 5943_-1 (0.809)

## Transaction row 542000 (TransactionID 3529000)
- `V288`: 0.754
- `V70`: 0.737
- `D10`: 0.716
- `V40`: 0.710
- `C7`: 0.710
- `freq_P_emaildomain`: 0.700
- `C2`: 0.697
- `V304`: 0.695
- `C13`: 0.687
- `D8`: 0.684
- edge `rev_owns`: 301446 -> 9633_-1 (0.850)
- edge `uses_device`: 214250 -> SM-G920P Build/NRD90M (0.822)
- edge `rev_owns`: 397273 -> 9633_-1 (0.811)
- edge `uses_device`: 541385 -> SM-G920P Build/NRD90M (0.809)
- edge `rev_owns`: 137783 -> 9633_-1 (0.808)

## Transaction row 553434 (TransactionID 3540434)
- `V20`: 0.725
- `V288`: 0.724
- `freq_M9`: 0.701
- `D5`: 0.700
- `V40`: 0.695
- `hour_sin`: 0.695
- `freq_M2`: 0.694
- `C7`: 0.691
- `freq_P_emaildomain`: 0.686
- `V87`: 0.686
- edge `rev_owns`: 227152 -> 9633_-1 (0.849)
- edge `uses_device`: 283670 -> SM-G920P Build/NRD90M (0.821)
- edge `rev_owns`: 256838 -> 9633_-1 (0.810)
- edge `uses_device`: 372660 -> SM-G920P Build/NRD90M (0.809)
- edge `rev_owns`: 127240 -> 9633_-1 (0.809)

## Transaction row 553431 (TransactionID 3540431)
- `V288`: 0.731
- `D10`: 0.707
- `V20`: 0.702
- `C7`: 0.702
- `D5`: 0.701
- `freq_M9`: 0.699
- `V40`: 0.691
- `V105`: 0.684
- `V30`: 0.682
- `C2`: 0.679
- edge `rev_owns`: 66108 -> 14076_-1 (0.850)
- edge `uses_device`: 390225 -> SM-G920P Build/NRD90M (0.821)
- edge `rev_owns`: 537727 -> 14076_-1 (0.811)
- edge `uses_device`: 541434 -> SM-G920P Build/NRD90M (0.809)
- edge `rev_owns`: 358440 -> 14076_-1 (0.808)

## Transaction row 538595 (TransactionID 3525595)
- `V288`: 0.750
- `freq_P_emaildomain`: 0.716
- `V304`: 0.708
- `V44`: 0.706
- `freq_M9`: 0.704
- `V280`: 0.690
- `hour_sin`: 0.690
- `V310`: 0.687
- `D8`: 0.682
- `D1`: 0.682
- edge `rev_owns`: 512401 -> 10568_-1 (0.849)
- edge `uses_device`: 478140 -> LG-M700 Build/NMF26X (0.822)
- edge `rev_owns`: 359364 -> 10568_-1 (0.811)
- edge `uses_device`: 408726 -> LG-M700 Build/NMF26X (0.809)
- edge `rev_owns`: 122217 -> 10568_-1 (0.809)

## Transaction row 565918 (TransactionID 3552918)
- `V29`: 0.684
- `V100`: 0.680
- `C7`: 0.678
- `freq_addr1`: 0.678
- `V96`: 0.677
- `V75`: 0.675
- `V30`: 0.673
- `D7`: 0.672
- `V289`: 0.670
- `V288`: 0.666
- edge `rev_owns`: 214970 -> 1722_-1 (0.850)
- edge `uses_device`: 151898 -> Windows (0.822)
- edge `rev_owns`: 251855 -> 1722_-1 (0.810)
- edge `uses_device`: 26165 -> Windows (0.809)
- edge `rev_owns`: 301587 -> 1722_-1 (0.809)

## Transaction row 538594 (TransactionID 3525594)
- `V288`: 0.741
- `V280`: 0.715
- `D5`: 0.710
- `V44`: 0.709
- `V304`: 0.699
- `V49`: 0.696
- `freq_P_emaildomain`: 0.694
- `freq_M9`: 0.690
- `V130`: 0.677
- `V45`: 0.671
- edge `rev_owns`: 529627 -> 10568_-1 (0.846)
- edge `uses_device`: 527163 -> LG-M700 Build/NMF26X (0.821)
- edge `rev_owns`: 401018 -> 10568_-1 (0.811)
- edge `uses_device`: 448385 -> LG-M700 Build/NMF26X (0.810)
- edge `rev_owns`: 125937 -> 10568_-1 (0.809)

## Transaction row 538592 (TransactionID 3525592)
- `V288`: 0.758
- `freq_M9`: 0.735
- `V30`: 0.683
- `V36`: 0.678
- `D1`: 0.677
- `V76`: 0.675
- `V126`: 0.673
- `D9`: 0.671
- `V302`: 0.666
- `V38`: 0.665
- edge `rev_owns`: 9906 -> 10568_-1 (0.848)
- edge `uses_device`: 106931 -> LG-M700 Build/NMF26X (0.822)
- edge `rev_owns`: 435674 -> 10568_-1 (0.811)
- edge `uses_device`: 487289 -> LG-M700 Build/NMF26X (0.809)
- edge `rev_owns`: 129595 -> 10568_-1 (0.809)

## Transaction row 536870 (TransactionID 3523870)
- `freq_M9`: 0.735
- `V30`: 0.691
- `V38`: 0.689
- `V76`: 0.683
- `D15`: 0.667
- `D1`: 0.666
- `V36`: 0.665
- `V43`: 0.662
- `V292`: 0.657
- `V295`: 0.654
- edge `rev_owns`: 103432 -> 10568_-1 (0.861)
- edge `rev_owns`: 532245 -> 10568_-1 (0.815)
- edge `rev_owns`: 148310 -> 10568_-1 (0.811)
- edge `uses_email`: 281107 -> gmail.com (0.811)
- edge `uses_email`: 368574 -> gmail.com (0.804)

## User 4663_330 (account-risk head)
- edge `rev_owns`: 63656 -> 4663_330 (0.875)
- edge `rev_owns`: 123052 -> 4663_330 (0.860)
- edge `owns`: 4663_330 -> 193599 (0.836)
- edge `owns`: 4663_330 -> 97351 (0.820)
- edge `owns`: 4663_330 -> 2 (0.818)

## User 8431_251 (account-risk head)
- edge `rev_owns`: 274919 -> 8431_251 (0.876)
- edge `rev_owns`: 160 -> 8431_251 (0.860)
- edge `owns`: 8431_251 -> 166536 (0.834)
- edge `rev_owns`: 334641 -> 8431_251 (0.821)
- edge `owns`: 8431_251 -> 12181 (0.820)

## User 13276_126 (account-risk head)
- edge `rev_owns`: 217452 -> 13276_126 (0.877)
- edge `rev_owns`: 410225 -> 13276_126 (0.854)
- edge `owns`: 13276_126 -> 321573 (0.836)
- edge `owns`: 13276_126 -> 83310 (0.821)
- edge `owns`: 13276_126 -> 347519 (0.820)
