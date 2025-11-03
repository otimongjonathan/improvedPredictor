# Agricultural Cost Predictor - Performance Summary

## Model Performance Metrics

### R² Scores (Coefficient of Determination)

| Cost Type | R² Score | Interpretation |
|-----------|----------|----------------|
| **Seed** | 66.63% | Moderate |
| **Fertilizer** | 63.38% | Moderate |
| **Herbicide** | 62.25% | Moderate |
| **Pesticide** | 62.60% | Moderate |
| **Labor** | 59.64% | Fair |
| **TOTAL COST** | **80.81%** | **Very Good** |

### Test Set Metrics

#### Individual Costs (Normalized Scale)
- **Seed**: RMSE 0.5855, MAPE 154.04%
- **Fertilizer**: RMSE 0.5767, MAPE 98.99%
- **Herbicide**: RMSE 0.6148, MAPE 91.77%
- **Pesticide**: RMSE 0.5827, MAPE 87.19%
- **Labor**: RMSE 0.6316, MAPE 177.24%

#### Total Cost (Actual Scale)
- **MSE**: 20,744,332.37
- **MAE**: 3,640.12 UGX
- **RMSE**: 4,554.59 UGX
- **MAPE**: 6.08%
- **R²**: **80.81%**

### Training Summary
- **Training Sequences**: 4,377
- **Validation Sequences**: 240
- **Test Sequences**: 240
- **Best Model**: Epoch 48 (Validation Loss: 1.1541)
- **Final Validation Loss**: 1.3147
- **Model Parameters**: 104,645

## Model Architecture
- **Hybrid Model**: Temporal LSTM + Graph Attention Network (GAT)
- **Temporal Input**: 12-sequence length with 8 features
- **Graph Nodes**: 240 (unique Region-District-Crop combinations)
- **Graph Edges**: 17,760

## Key Achievements
1. ✅ **Total Cost Prediction**: 80.81% R² - Excellent performance for agricultural cost prediction
2. ✅ **Training, Validation, and Testing**: All three phases completed successfully
3. ✅ **Model Saved**: Best and final models saved for production use
4. ✅ **Preprocessor Saved**: Data preprocessing pipeline saved for consistent predictions

## Interpretation
- The model performs **very well** for total cost prediction (80.81% variance explained)
- Individual cost predictions have moderate performance (60-67% R²), which is acceptable given the complexity of agricultural input pricing
- The high total cost R² suggests the model captures the overall cost structure effectively

