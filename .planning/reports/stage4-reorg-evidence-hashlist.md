# stage4-reorg 研究产出 61 件逐件哈希清单（evidence）

> 用途：阶段 4 目录搬映第 2 项（`Wealth_Management_data/研究产出/` → `results/研究产出_2026-10/`）的字节级留痕，供任何时点独立核对 61 件产物搬移零丢失零改写。
> **字节级复查源：`Wealth_Management_data.7z`（2026-10-06 21:44 快照，工作区根，LAST_WRITE=2026-10-06 21:44）**——从该快照解出 `研究产出/` 原件，逐件对拍下表 SHA256 即可复现搬移前字节。
> 口径：SHA256 = 文件字节（搬移前后逐件一致，61/61 对账 0 不一致）；git blob = 入库对象（仓库换行规整后的对象号，取前 12 位）。
> 搬移对账：源 61 件 78,450 字节 → 目标 61 件 78,450 字节；源残留 0；空目录骨架 10 个保留未删。
> 生成：2026-10-13（stage4-reorg 连单落档）。

| 相对路径（`results/研究产出_2026-10/` 之下） | 字节 | SHA256（文件字节） | git blob |
|---|---|---|---|
| audit_resid_adj.csv | 9624 | 3f3c41f1fae871d880c65d25c9d22ca891fdad9cf70174987a01f4cd97392e2d | 88361d3db148 |
| factor_scan.csv | 1858 | 14af8733d599f74a93b492209e768580aeb7812ac5a63946a53deb84cde69d6f | 6d5ec82cf8d1 |
| factor_study_full/ic_amount20.csv | 479 | 184478fdedd7c616e58eb5ad0d7a3bc7539ce49f315c3a1c5692cd2a5265fcf2 | 721ffba9cd99 |
| factor_study_full/ic_downvol60.csv | 482 | 13db463df36d9314fce18aebc42f85091e0ffa2e394a551cc49dee333d037fe5 | 0181afdc137f |
| factor_study_full/ic_hhhl20.csv | 486 | 32e41872398774f17f33e4deb7c135cd3150421c5478c8772aee98a2a7fdb050 | 7f2e7423876a |
| factor_study_full/ic_mom120_skip20.csv | 485 | 18b20b731418c797bb55b89f16d86191b425bb4dd5dd83768e1fd89959f109c7 | 05c91fc8ef0a |
| factor_study_full/ic_mom20.csv | 483 | 4ebc4417029a6dfbe4d4e1ef8c2053266e43b5630493d12070c62a7e71358c0e | 73af7c1d4f28 |
| factor_study_full/ic_mom60_skip20.csv | 487 | 50aef0c6d79e43b123d0933457f5e5c7416f16a0c672a4086e00d0f9f002fe22 | f8e2cbad2ddd |
| factor_study_full/ic_pos250.csv | 481 | 1031daec6ff2f00884ec9a72ce7d792580f6a6cec143a5cc88a0690c33f27ad9 | ba1021110b75 |
| factor_study_full/ic_rev20.csv | 468 | ef567fd37b11bacf8d85b55de319412f2be2bb711a400c472560e6ed67a8d2df | 74b21864b699 |
| factor_study_full/ic_rev5.csv | 467 | 3a290686da013d7f9183d7ccce48e693340fcd942a2a7932aa71792bc83ccf06 | b9de0b8c7db0 |
| factor_study_full/ic_vol60.csv | 483 | 77d01e6276324837aa9d8002d954eb70adbef4126cf49ae203362258d61eb28f | 2ab4e8f57d8f |
| factor_study_full/ic_volratio5_60.csv | 475 | 57c8fa84ac9cbc6b8a93f6b6e507977b3aff394ff70b17592144fdca2e12bd6f | 55d290da4772 |
| factor_study_full/run_meta.json | 261 | 13ea4927eeb057545b0ef6bdd7e7eb2bc45136bc21162c7e2bb007ff26413480 | 1f679a3e0296 |
| factor_study_full/subperiod_ic.csv | 4412 | abdbc771c6d5338dde27ec37bcaac22d3cd789182d3cd36f602f7061d8d5311f | a48a28bb671c |
| factor_study_full/summary.csv | 1377 | 2e5cf8305d961e00fc056b80baa6ecce3b8664374a6373db13dfbb865c428e1f | 8f6e0af688df |
| financial/ic_bp.csv | 465 | a34ebaa0bf78282ceedca3dab04bf98a263dc4417b1cd07c3c843ad076385c0b | 745add3dea9a |
| financial/ic_cf_quality.csv | 478 | b5509df50a415a3d6a9fd04e64e866ceb90af2da6b8b56cbe2b6cafb852886cb | 4c8e7f0995c2 |
| financial/ic_ep.csv | 475 | 6ce6f51531455a72866f45e644083124e5c3ac7e424fd2a88f4d7ba8026e4d66 | 6c7955b33123 |
| financial/ic_gross_margin.csv | 489 | bed0de543f119046070a37dbc69a7f7aa69b2c1192616899c947a1ba625846d1 | 5c6a151cb017 |
| financial/ic_profit_yoy.csv | 479 | 4541d9b7975d1ee9909d59624e8dd4ba9c62c9e084938b21b370cd2e093dd6b9 | f6b81ed5493f |
| financial/ic_revenue_yoy.csv | 478 | f4bb55c645ad885138238335ab863b9a03d736e79aba6e7d2ebba5f4e07e8670 | 5802c8eb0bf9 |
| financial/ic_roe.csv | 476 | 8e59fa5be0a07ce6e476ba931c63cd3ab7669360d0f41f71c226a1d6d62ebf96 | aebcd7c49392 |
| financial/subperiod_neutral.csv | 2788 | 8012c7eee6280b0efcec39a0d796b7b0c4376b864b4b73c4becd7009d431bd84 | 8ec596c2dc73 |
| financial/subperiod_raw.csv | 2883 | 5291bdf74bb08649b342d5321288f4f7a12d6077c22c87ae980de63d3dc71fc5 | 2b8d82f9ea44 |
| financial/summary_neutral.csv | 1024 | 50e656aabb44ce4144a0889ef4c139335143d2d1346002d646c050cb8ca3aee7 | 571ed9ce537f |
| financial/summary_raw.csv | 1072 | 3d6d9adad5f7ec720e1c0ff59a6503e39663e727405f4aec1b2830dbda9f5cf4 | dc950999c70c |
| headtohead/vs_qlib.csv | 1158 | cd0be596ec9329d6268df4f80bfa869b0baf271921d2c8fc56bdb76028ee9bdf | 4a8c1116ad12 |
| holding/holding_period_test.csv | 2189 | 67a18664afbe4111e8df9ef2426e97f6363fbc8f5c3c718ff87071adbc71a75f | ef355741eb53 |
| holding/ic_multi_period.csv | 1629 | 3b3507d05fe1f4ecb5115bac625e1f87a87a035c0881e184173b50593a6faac2 | d63e7e7ffed9 |
| ic_downvol60.csv | 482 | 5c8812537332956f0e120eebc7448beace836529064a2f43909daec0c766bba8 | 4955b9bb1f13 |
| ic_hhhl20.csv | 484 | 2c351c1a3e12cf47684fd7b9c7bf015151dabc603a352420eec9c8af0f7d322f | b9389b1172c5 |
| ic_mom120_skip20.csv | 485 | 6f215cb25e64fcf4022b72ef54d848961b5dea811eaf604fd9f572586d9870a8 | a349a2535005 |
| ic_mom60_skip20.csv | 483 | 5e0809e114828bb2e60e679b7b64f73bfa3c289881796eaf722c315a4f7da4fc | 689b39a99e1d |
| ic_pos250.csv | 484 | 27dbcf704fa717f0dbe7cd64a4d8af8f6293381f283679e192ffeda50eee8931 | 16ea46053a52 |
| ic_rev5.csv | 460 | f78193e78786af953172180296be1f7eeefe16e5593bd66bcf46f77abdb60820 | 5422e5ac54fa |
| ic_vol60.csv | 480 | 697e80c4dd463ebe8fd74ad1a5f4b7668762da79f0451d157a10729dd38132bb | a94111c74d37 |
| long_only/factor_inout.csv | 343 | f2910d73dee9fdce31c077136c282c2416c1032d56b2ef1051f0d461cdcae56b | 64c3762193f0 |
| long_only/param_sweep.csv | 1348 | 19ae37cea8f99327f57fb9efaa790610db6012f70c050725f394a1b73059470d | b0dcca4bfc94 |
| long_only/run_meta.json | 299 | f89ce378406611ef52dd89aa01870424701b06b0329b7c9e17e51f8971f1b31d | 89b7ab35ef87 |
| probe_all.json | 9644 | d4d3509b449dc1f9caf1bd27c17e2db8cdded09a23e7e8fa318941da5cd59270 | 934c79a7e9ff |
| run_meta.json | 198 | 11a446562933dfe65859b72c6055e1cddf5247590447c4a172148bed2d5be0e2 | 566b66a39859 |
| scorecard_full/scorecard.csv | 7824 | b67d3f4a3c738138dcbaa08e26a36323bf3a4adfa3e870c50903805edcb3911c | e80730b9212a |
| subperiod_ic.csv | 280 | 0d093f237beeee5463e800b4ae8dbdd31fafdd0b8f13b8b1ae0a94f5fcaa5e90 | 0ae7490534f9 |
| summary.csv | 192 | ee64fe72823861e02883e19369b5c2cb5da2668c19b281d97bcf499da051b556 | 65e0b98ba00c |
| synth_pair/synthesis_compare.csv | 1357 | 076d8ab06b265cd67a130822536c2b5fcc612dfcebabdd4bc2f62c7b3a47ac32 | 642c0e5486b8 |
| turnover/turnover_scan_low_vol_120-bp-cf_quality.csv | 1481 | 760c8ad0f28ac64aafb316d1422144434ce4e456a0425b4c9738064e4d35675b | ea79d4b8e112 |
| walk_forward/run_meta.json | 261 | 230c6d78de8886758106dabfddb88cb80d9a20868b3f39b5aaf0d5b95441a38e | 2f2b3070709d |
| walk_forward/size_buckets.csv | 1558 | f748fb93bbf50773fbc6cd74015e088476e25a33278595463e17c5b855afeedc | 1fd53b5dcd9f |
| walk_forward/summary.csv | 780 | cc1c2fc655833287a508336bb2a4fc053274c7725f50c176618a68b09ef50763 | 7ee19c8b7ba4 |
| walk_forward/windows_amount20.csv | 968 | f4329a20c9fb21cd59a05ef00caab8bbf9185a7be9d99b00602c8a5cb9828a5d | 14a431270fd2 |
| walk_forward/windows_downvol60.csv | 964 | 804e9daffc4662ba700aa4d61b4bf72bff238333b9a618f51beb7569b50a5efe | 12390dfd9e8f |
| walk_forward/windows_hhhl20.csv | 973 | eb56b91dbb1df09c6da7239d0ab79b8e2c4251073873c5eba04d7f3845009145 | da23c4b69ecb |
| walk_forward/windows_mom120_skip20.csv | 971 | bf4fd1e76200b0b037e653c202ae698f1e02e07aadf0196611955202e81ea24f | 36e4f3eb140d |
| walk_forward/windows_mom20.csv | 967 | 64d0d0ff8bb602ab91450b1aec270631cb2fb1f6271dec16aaab4e5243d45d54 | 327e5af5f785 |
| walk_forward/windows_mom60_skip20.csv | 965 | 4bcb8a4f24ebf2edabb7d4ea47e68240871a71b51841182f37453e1485740d8d | 6c16fca9a7e4 |
| walk_forward/windows_pos250.csv | 964 | 29d255d3abeb35b189467954912202f17eeb50b13183cf9f92e6283e729ade64 | e1e227ed3256 |
| walk_forward/windows_rev20.csv | 954 | 81613942504211a51689b06888b8a55cc011784660963903579b70916a1948c1 | 6172c29a1b39 |
| walk_forward/windows_rev5.csv | 969 | d484633b69e7c1b3b954ffd45a4f3b7e25ca3e15e1b42b0b667459e320ee4ba3 | 6b0669b12f8f |
| walk_forward/windows_vol60.csv | 973 | e4f5a5474f92c68b59d8c924f47623a5671d35c1920ee2a958356925ab7fd726 | 988bcd50300e |
| walk_forward/windows_volratio5_60.csv | 968 | dd1242393cdfc8596cb149b3f4a7ba320d185359fc24200210c85d612fa13566 | 2294f480e979 |
合计 **61 件 / 78,450 字节**（与 21dbe49 提交清单、git show --stat "61 files changed" 三方一致）。

不构成投资建议。