# Contributing to Plant Leaf Classification

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## 📋 Code of Conduct

Please be respectful and constructive in all interactions with other contributors and maintainers.

## 🚀 How to Contribute

### 1. Reporting Bugs

If you find a bug, please create an issue with:
- **Title**: Clear description of the bug
- **Description**: Detailed explanation of the issue
- **Steps to Reproduce**: How to reproduce the bug
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: Python version, OS, and dependency versions

### 2. Suggesting Enhancements

To suggest a new feature or enhancement:
- **Title**: Brief description of the suggestion
- **Description**: Detailed explanation with use cases
- **Justification**: Why this feature would be useful
- **Implementation**: Your ideas on how it could be implemented (optional)

### 3. Submitting Pull Requests

#### Step 1: Fork and Clone
```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/yourusername/plant-leaf-classification.git
cd plant-leaf-classification
```

#### Step 2: Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/bug-description
```

#### Step 3: Make Your Changes
- Write clean, readable code
- Follow PEP 8 style guidelines
- Add comments for complex logic
- Update docstrings if needed

#### Step 4: Test Your Changes
```bash
# Run the project to ensure it still works
python model.py
python predict.py
python cbir.py
```

#### Step 5: Commit Your Changes
```bash
git add .
git commit -m "Descriptive commit message"
# Examples:
# git commit -m "Add feature: new model comparison chart"
# git commit -m "Fix: resolve memory leak in feature extraction"
# git commit -m "Docs: update README with usage examples"
```

#### Step 6: Push to Your Fork
```bash
git push origin feature/your-feature-name
```

#### Step 7: Create a Pull Request
- Go to GitHub and create a PR from your fork to the main repository
- Provide a clear title and description
- Reference any related issues (#issue-number)
- Wait for review and feedback

## 📝 Contribution Guidelines

### Code Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use meaningful variable and function names
- Write docstrings for all functions
- Keep lines under 100 characters

### Commit Messages
- Use present tense: "Add feature" not "Added feature"
- Be descriptive: "Fix image loading error in CBIR module" not "Fix bug"
- Keep first line under 50 characters
- Add detailed description in the body if needed

### Documentation
- Update README.md if you add new features
- Document new functions with docstrings
- Add usage examples for new features
- Update requirements.txt if adding dependencies

### Testing
- Test your changes locally before submitting
- Verify that all existing functionality still works
- Test edge cases and error handling

## 🎯 Areas for Contribution

### Code Improvements
- Optimize feature extraction speed
- Improve model accuracy
- Add new ML algorithms
- Optimize memory usage
- Add parallel processing

### Features
- Add support for more image formats
- Implement deep learning models
- Add web interface
- Create Docker container
- Add real-time prediction from webcam

### Documentation
- Improve README clarity
- Add tutorial notebooks
- Create video tutorials
- Translate documentation

### Testing
- Add unit tests
- Add integration tests
- Improve error handling
- Create test datasets

## 📞 Getting Help

- **Questions?** Open an issue labeled `question`
- **Need Guidance?** Comment on an issue or PR
- **Have Ideas?** Create a discussion thread

## ✅ Review Process

1. **Automated Checks**: Your PR will run basic checks
2. **Code Review**: A maintainer will review your code
3. **Feedback**: You may receive suggestions for improvements
4. **Approval**: Once approved, your PR will be merged

## 🎉 Thank You

Your contributions make this project better for everyone. Thank you for being part of the community!

---

**Happy Contributing! 🚀**

Questions? Feel free to reach out or open an issue.
