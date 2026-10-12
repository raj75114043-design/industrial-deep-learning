# Evaluation method

- **ANN:** 3,000 batches split 70 / 15 / 15; lab errors screened on train and validation, removed from test; 500 new-supplier batches as an out-of-distribution set. Noise floor = RMSE of the true (noise-free) dissolution.
- **CNN:** 4,000 train, 2,000 validation (temperature and τ), 3,000 test at natural class frequencies; 2,000 line-2 images; 600 + 1,000 labelled line-2 images for adaptation and its validation; 600 line-1 images replayed during fine-tuning.
- **RNN:** 120 training and 30 validation units run to failure; 100 test units truncated at a random point (the fleet in service). RMSE and NASA score on each unit's last window; early warning on all test windows.
- Seeds are fixed; gates have margins because CPU training is not bit-identical across machines.
