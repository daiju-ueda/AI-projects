# Private
個人用のレポジトリ

# 導入
## 必要環境
- Python 3.7
  - `brew install 'python@3.7'`
- [Pipenv](https://pipenv-ja.readthedocs.io/ja/translate-ja/)(導入手順は下記参照)

### Pipenvについて

pythonバージョンとpip環境を Pipenv, Pipenv.lock というシンプルなテキストファイルで管理できるpython仮想環境管理ツールです。
利用方法は [Pipenvを使ったPython開発まとめ](https://qiita.com/y-tsutsu/items/54c10e0b2c6b565c887a) が参考になります。

mac なら Homebrew で `brew install pipenv` でインストールできます。
```shell
brew install pipenv

# 利用時
pipenv shell など
```

または、 pip (Python3以上)を使ってインストールします。
```shell
# インストール(pip3の部分は環境によって読み替えてください)
pip3 install pipenv

# 利用時
pipenv shell など
```
環境によってpipenvコマンドができていない場合、pipenv の代わりに `python3 -m pipenv shell` のようにモジュール呼び出し形式でも利用できます。


## セットアップ
プロジェクトフォルダ内(Pipenvがあるフォルダ以下)で、 `pipenv install` を行うと、各自のローカル環境に仮想python環境 と依存 pip がインストールされます。

`pipenv install` は Pipenv.lock ファイルを元に仮想環境を構築・更新するためのコマンドです。
Pipenv.lock を誰かが更新した場合などに随時行ってください。差分がなければ何もしないので、何度実行しても大丈夫です。

## pipenv環境の有効化
Pipenv ファイルのあるフォルダ配下で pipenv shell を実行します。
```bash
pipenv shell
```
または、リポジトリ内にある `./activate.sh` を実行でもOKです。

```bash
path/to/AI-projects/activate.sh
```

## pipenv にpipライブラリを追加する
開発環境のバージョン管理のため、pip install ではなく、pipenv install を使ってください。

Pipenv ファイルのあるフォルダ配下で pipenv install "pipライブラリ名" を実行します。
pipenvコマンドの実行後は Pipenv.lock や Pipenv ファイルが更新されるので、追加後にコミットしてください。
```bash
# pip ライブラリの numpy, Pillow をインストールする
pipenv install numpy Pillow
```
