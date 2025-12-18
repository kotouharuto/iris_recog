# 虹彩認証システム (Iris Recognition System)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

古典的な画像処理手法（OpenCV）とガボールフィルタを用いた、虹彩認証アルゴリズムの実装プロジェクトです。
前処理、瞳孔・虹彩の円検出、正規化（ラバーシートモデル）、特徴抽出、およびハミング距離によるマッチングまでの一連のパイプラインを提供します。

元々はGoogle Colab上で開発されていたものを、ローカル環境およびGitHub管理用に移植・構成したものです。

## 📖 概要 (Overview)

MMU Iris Database 等の眼球画像を入力とし、個人識別（1:N認証 または 1:1照合）を行います。
ディープラーニング（CNN等）を使用せず、幾何学的特徴とテクスチャ解析に基づく古典的アプローチ（Daugmanの手法に近い構成）を採用しています。

### 主な処理フロー
1.  **前処理 (Preprocessing)**
    * 反射光（specular reflection）の除去 (Inpainting)
    * CLAHEによるコントラスト強調
    * メディアンフィルタによる平滑化
2.  **領域分割 (Segmentation)**
    * 瞳孔検出: 閾値処理 + 輪郭抽出 (FindContours)
    * 虹彩検出: Cannyエッジ検出 + Hough変換 (HoughCircles)
3.  **正規化 (Normalization)**
    * ドーナツ状の虹彩領域を極座標展開し、固定サイズ（64x512）の短冊画像に変換（Rubber Sheet Model）
4.  **特徴抽出 (Feature Extraction)**
    * 2D Gabor Filter を用いてテクスチャの位相情報を抽出
    * 実部・虚部を量子化し、Iris Codeを生成
5.  **マッチング (Matching)**
    * マスク処理付きハミング距離計算
    * 頭の傾きや撮影ズレを考慮した回転補正（Bit Shift Matching）

## 📂 ディレクトリ構造 (Directory Structure)

```text
iris-recognition/
├── data/                  # データセットディレクトリ (Git管理対象外)
│   └── MMU-Iris-Database/ # 画像データはここに配置
├── src/                   # ソースコード
│   ├── config.py          # パラメータ設定
│   ├── preprocessor.py    # 画像処理・正規化クラス
│   ├── feature_extractor.py # 特徴抽出クラス
│   ├── matcher.py         # マッチング計算クラス
│   └── utils.py           # データロード・可視化ツール
├── main.py                # 実行用エントリーポイント
├── requirements.txt       # 依存ライブラリ一覧
└── README.md              # 本ドキュメント
```

## ⚙️ 環境構築 (Installation)

本プロジェクトを実行するためのローカル環境セットアップ手順です。
他のプロジェクトとのライブラリ競合を防ぐため、**仮想環境 (venv)** の使用を強く推奨します。

### 1. 前提条件 (Prerequisites)
* Python 3.8 以上がインストールされていること
* Git がインストールされていること

### 2. リポジトリのクローン
```bash
git clone [https://github.com/](https://github.com/)[あなたのユーザー名]/iris-recognition.git
cd iris-recognition
```

### 3. 仮想環境の作成と有効化 (Virtual Environment)

OSに合わせて以下のコマンドを実行し、仮想環境を作成・有効化してください。

**Windows (PowerShell):**
```powershell
# 仮想環境 'venv' を作成
python -m venv venv

# 仮想環境を有効化
.\venv\Scripts\activate

# (成功すると行頭に (venv) と表示されます)
```

**Mac / Linux:**
```bash
# 仮想環境 'venv' を作成
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate

# (成功すると行頭に (venv) と表示されます)
```

### 4. 依存ライブラリのインストール
仮想環境が有効になっている状態で、ライブラリを一括インストールします。

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** GPU (CUDA) を使用する場合は、`requirements.txt` のインストール後に、ご自身の環境に合った PyTorch を [公式サイト](https://pytorch.org/get-started/locally/) から手動でインストールし直す必要がある場合があります。

### 5. データセットの配置
本リポジトリには画像データは含まれていません。
[MMU Iris Database](https://www.kaggle.com/datasets/naureenmohammad/mmu-iris-dataset) などをダウンロード・解凍し、以下のように配置してください。

## 🚀 使い方 (Usage)

以下のコマンドで、データセットの読み込みから前処理、特徴抽出、認証実験（EER算出）までを一括で実行します。

```bash
python main.py
```

### 実行オプション
（※ `main.py` に引数処理を実装した場合の記述例）
```bash
# 特定の画像のみ処理を確認する場合
python main.py --mode debug --image ./data/MMU-Iris-Database/1/left/aleft.bmp
```

## 🛠️ パラメータ設定 (Configuration)

`src/config.py` にてアルゴリズムの挙動を調整可能です。

* `NORM_HEIGHT` / `NORM_WIDTH`: 正規化画像の解像度 (Default: 64x512)
* `SHIFT_RANGE`: マッチング時の回転許容ビット数 (Default: 8)
* `CLAHE_CLIP`: 前処理のコントラスト強調度合い
* `GABOR_PARAMS`: ガボールフィルタの波長や方向設定

## 🤝 開発ガイドライン (Development Guidelines)

### 🌿 ブランチ運用ルール (Branching Strategy)

本プロジェクトでは、シンプルで管理しやすい **GitHub Flow** をベースにした運用を行います。

| ブランチ名プレフィックス | 用途 | 例 |
| :--- | :--- | :--- |
| `main` | **安定版**。常に動作保証されたコードのみを含める。 | `main` |
| `feature/` | 新機能の開発。 | `feature/add-clahe-preprocessing` |
| `fix/` | バグ修正。 | `fix/normalization-index-error` |
| `experiment/` | **実験・検証**。パラメータ変更や新手法のトライアル用。<br>（結果が良ければ `main` にマージ、悪ければ破棄） | `experiment/gabor-wavelength-tuning` |

### 📝 コミットメッセージ (Commit Messages)

変更内容を明確にするために **Gitmoji** を採用しています。
コミットメッセージの先頭に、以下のルールに従って絵文字を付与してください。

| 絵文字 | プレフィックス | 意味・用途 (Meaning) |
| :---: | :--- | :--- |
| ✨ | `feat` | 新しい機能の実装 (New feature) |
| 🐛 | `fix` | バグ修正 (Bug fix) |
| 🧪 | `experiment` | パラメータ調整、実験設定の変更 (Experiment / Tuning) |
| ♻️ | `refactor` | 機能を変えずにコードを整理 (Refactoring) |
| 📝 | `docs` | ドキュメントの修正・追加 (Documentation) |
| 🍱 | `data` | データセット、画像の追加・変更 (Assets / Data) |
| ⚡️ | `perf` | パフォーマンス・速度改善 (Performance) |
| 🎨 | `style` | コードフォーマット、空白、セミコロン等の修正 (Code style) |
| ✅ | `test` | テストコードの追加・修正 (Tests) |
| 🚧 | `wip` | 作業中 (Work in progress) |

**記述例:**
* `✨ feat: CLAHEによる前処理機能を追加`
* `🐛 fix: ハミング距離計算時のゼロ除算エラーを修正`
* `🧪 experiment: 閾値を20から25に変更して精度検証`

## 👤 著者 (Author)

* [あなたの名前]
* 所属: [所属大学・組織名など]

## 📜 ライセンス (License)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.