# Poetry Generation using LSTM (Tensorflow, Keras)
This project implements a poetry generation model trained on combined lyrics/texts from multiple artists using an LSTM-based neural network

## 🗂️ Dataset
- Text files combined from:
  - Adele, Bob Marley, Lady Gaga, Kanye West, Eminem
- All texts merged into `dataset/MERGED.txt`

## 📚 Technologies & Libraries
- Python 3.12.4
- Tensorflow / Keras
- pandas, numpy
- string, request (standard libraries)

## 🔧 Model Overview
- Tokenization of input text
- Prepare sequences of increasing length for training (next-word prediction)
- Padding sequences to fixed length
- LSTM model with two layers
- Dense layers for final classification over vocabulary
- Categorical crossentropy loss with Adam optimizer


## 🤖 Model Training
- Vocabulary size determined from tokenization
- Input sequences padded to length 20
- Batch size: 32
- Epochs: 100

## 🎭 Poetry Generation
- Function `generate_poetry(seed_text, n_lines)` generates `n_lines` lines starting from `seed_text`
- Predicts next words iteratively using trained model
- Prints generated lines

## ⚙️ Setup Instructions
```bash
git clone https://github.com/troyhunterz/poetry-generation.git
cd poetry-generation
```
- Place lyric text files into `dataset` folder
- Run training script
- Use `generate_poetry` function to produce new lines

## 🧾 License
This project is licensed under the MIT License.

## 👤 Author
troyhunterz

email: ann0nfolder@gmail.com