# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-29

### Added
- Initial project release
- Model training with 4 ML algorithms (DecisionTree, RandomForest, SVM, KNN)
- Automatic best model selection based on accuracy
- Feature extraction (color, texture, shape)
- Model visualization (confusion matrix, classification report, performance charts)
- K-Fold cross-validation training curves
- Single image prediction module
- Content-Based Image Retrieval (CBIR) module
- Sensor data fusion capability (optional)
- Comprehensive documentation (README, CONTRIBUTING, CHANGELOG)
- Requirements file for easy dependency installation

### Features
- **model.py**: Train and evaluate multiple ML models on leaf dataset
- **predict.py**: Predict class of single leaf image
- **cbir.py**: Find top-3 similar images using cosine similarity
- Automatic artifact generation (models, scalers, visualizations)
- Progress bars for dataset loading

### Technical Details
- Python 3.11+ support
- Scikit-learn machine learning library
- OpenCV for image processing
- K-Fold validation (6 folds)
- 80/20 train-test split with stratification

### Performance
- RandomForest Model Accuracy: 100%
- Macro Precision: 1.0000
- Macro Recall: 1.0000
- Macro F1-Score: 1.0000

### Dataset
- 207 total images
- 4 classes (FN, K, N, P)
- 537-dimensional feature vectors per image

---

## Future Improvements (Roadmap)

### v1.1.0 (Planned)
- [ ] Add deep learning models (CNN)
- [ ] Web interface for predictions
- [ ] Real-time webcam prediction
- [ ] Batch prediction mode
- [ ] Confidence scores for predictions

### v1.2.0 (Planned)
- [ ] Docker containerization
- [ ] REST API for predictions
- [ ] Database integration for storing predictions
- [ ] Model retraining pipeline
- [ ] Unit tests

### v2.0.0 (Planned)
- [ ] Mobile app (React Native)
- [ ] Transfer learning with pre-trained models
- [ ] Support for new leaf categories
- [ ] Advanced visualization dashboard
- [ ] Multi-model ensemble predictions

---

## Notes

### Version 1.0.0 Highlights
- Achieved 100% accuracy on test set with RandomForest
- Implemented three-type feature extraction (color, texture, shape)
- K-Fold validation ensures model robustness
- Modular code structure for easy extension

### Known Limitations
- Small dataset size (207 images)
- Limited to 4 classes
- Requires image preprocessing (resizing to 224x224)
- No real-time inference optimization

### Dependencies
- opencv-python>=4.12.0.88
- scikit-learn>=1.7.2
- pandas>=2.3.3
- numpy>=2.3.5
- joblib>=1.5.2
- tqdm>=4.67.1
- seaborn>=0.14.0
- matplotlib>=3.9.0

---

## How to Report Changes

When you find bugs or improvements, please:
1. Open an issue with clear description
2. Include error messages and traceback
3. Specify your environment (OS, Python version)
4. Provide steps to reproduce (for bugs)

---

## Support

For questions or issues about the changelog:
- Open an issue on GitHub
- Check existing issues first
- Provide detailed information

---

**Last Updated**: April 2026  
**Version**: 1.0.0  
**Status**: Stable
