## イントロダクション
BCCDデータセット(血液の顕微鏡写真に血球のアノテーションを付加したデータセット)を使用して、白血球を検出するモデルを作成します。<br>
CNNフレームワークはpytorchを使用します。<br>
<br>

## 動作概要
- データを学習用(train)・検証用(val)に分割する
- モデルを生成する
- 学習用データで学習を行う
- 以下の学習スコアを表示する
    - 検出した物体のラベル推論のロス：　loss_c
    - 検出した物体のbounding bowの座標回帰結果のロス：　loss_l
- 学習済みモデルの重みデータの保存・読み込みを行う
- 検証用データから１つの画像を選んで物体検出（白血球WBC）を行う
<br>

使用方法はusage_train_test.ipynbを参照してください。<br>
<br>

## データの入手
BCCD_datasetディレクトリにて下記リポジトリをcloneしてください。<br>
https://github.com/Shenggan/BCCD_Dataset<br>
BCCDディレクトリ内のJPEGImagesディレクトリにJPEGファイルが、Annotationsディレクトリにbounding boxの情報が記載されたxmlファイルが格納されています。<br>
<br>

## 各モジュールの機能
- Dataset (utils/dataloader.py)
    - 入力：　画像ファイルのパス、bounding box除法データ(xml)のパス
    - 出力：　画像データとそれに属するbounding boxのラベル・座標
- Dataloader (utils/dataloader.py)
    - 画像データとそれに属するbounding boxのラベル・座標をバッチサイズぶん供給する
- gen_bccd_dataloader (utils/dataloader.py)
    - 入力：　検出対象とするクラス、バッチサイズ
    - 出力：　Dataloaderインスタンス
<br><br>

- SSD (model/cnn.py)
    - 物体検出モデル（SSD）
    - 学習済み推論モデルの重みデータの保存
    - 学習済み推論モデルの重みデータの読み込み
<br><br>

- Manager (utils/manager.py)
    - 物体検出モデルの学習
    - 学習スコアデータの作成・保存
    - 推論の実行
<br><br>

## Configファイルの設定内容 (config/config.py)
**ConfBCCD**<br>
| パラメタ名 | 内容 | default |
| ---- | ---- | ---- |
| LIST_LABEL | 物体検出の対象とするラベルのリスト | ["WBC"] |
<br>

**ConfDataloader**<br>
| パラメタ名 | 内容 | default |
| ---- | ---- | ---- |
| BATCH_SIZE | 学習バッチのサイズ（データ数） | 8 |
| SHUFFLE | エポック毎にデータの順序を入れ替える | True |
| TARGET_EXT | 対応可能な学習データファイルの拡張子 | [".jpg", ".jpeg"] |
<br>

**ConfTraining**<br>
| パラメタ名 | 内容 | default |
| ---- | ---- | ---- |
| LEARNING_RATE | 学習率 | 1e-4 |
| CLIP_GRAD_VALUE | gradientの最大値 | 2.0 |
| SGD_MOMENTUM | SDGのmomentum factor | 0.9 |
| SGD_WEIGHT_DECAY | SDGの正則化(L2)パラメタ | 5e-4 |
<br>

**ConfBoxDetector**<br>
| パラメタ名 | 内容 | default |
| ---- | ---- | ---- |
| CONF_TH | 検出したオブジェクトのクラスの推論結果のconfidenceの閾値。閾値以上であれば検出したとする。 | 0.8 |
| TOP_K | 検出結果のうち、クラス推論結果のconfidenceが高い順にいくつ使用するか | 100 |
| OVERLAP | default boxの中でオブジェクトを検出すべきものを抽出するIoU基準値 | 0.45 |
<br>

## System requirements (Dev environment)
- **OS:** Win10 (or latter)
- **Hardware** Intel CORE i5(8th gen)
- **Python:** 3.7 (or latter)
<br><br>

## Usage
usage_train_test.ipynbを参考にしてください<br>
<br>

## Misc
Copyright (c) 2022 SAri<br>
<br>


