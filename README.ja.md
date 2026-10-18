<p align="center">
  <img src="docs/assets/banner.jpg" alt="tsuzuri" width="100%">
</p>

<p align="center">
  <a href="docs/README.md">ドキュメント</a> ·
  <a href="examples/">サンプル</a> ·
  <a href="docs/evaluation.md">評価</a> ·
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  <a href="https://github.com/puresky271/tsuzuri/actions/workflows/ci.yml"><img src="https://github.com/puresky271/tsuzuri/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11 | 3.12 | 3.13">
  <img src="https://img.shields.io/badge/status-alpha-orange.svg" alt="Status: alpha">
  <img src="https://img.shields.io/badge/runs-offline%20by%20default-2ea44f.svg" alt="Runs offline by default">
  <img src="https://img.shields.io/badge/LLM-any%20OpenAI--compatible-6f42c1.svg" alt="Any OpenAI-compatible LLM">
  <img src="https://img.shields.io/badge/prompts-%E4%B8%AD%E6%96%87%20first-red.svg" alt="Prompts: Chinese first">
</p>

> [!NOTE]
> **tsuzuri は現在 alpha（0.1）です。** マイナーバージョン間で API が変わる可能性があります。プロンプトとテキストのヒューリスティクスは**中国語優先**です。他の言語でも保存と想起はできますが、抽出品質は中国語向けに調整されています。

tsuzuri（綴り）は、ユーザーが各キャラクターに話した内容を覚え、次の返答に関係するものを選び、プロンプトへ編み込みます。履歴を丸ごとコンテキストに詰め込まなくても、一週間後にキャラクターが「水曜の面接、どうだった？」と自然に聞けるようになります。

想定しているのは、複数の常駐キャラクター（エージェント）が一人のユーザーと一対一、またはグループで会話するプロダクトです。

- **エージェントごとの階層型記憶**：working → episodic → semantic、加えて procedural ルールと、全員で共有する *collective* スコープ。書き込みはルールで計画され（LLM がテーブルへ直接書き込むことはありません）、想起はハイブリッド検索（語彙／FTS、任意の密・疎ベクトル、構造化スロット、リランク、MMR）と ACT-R 風の減衰で行います。
- **ターンファクト（TurnFact）**：各やり取りを小さなカード（事実・状態変化・約束・訂正・未完の話題）に要約し、セッションをまたいで引き継ぎ、古い主張を上書きします。
- **コンテキストコンパイラ**：`compile_fact_workspace` などが、現在の状態、関連する事実、想起した記憶、会話の流れ、必要に応じた証拠、そしてモデルの注意が最も向く位置に置く「今回の実行パケット」を選別・重複排除・予算管理します。
- **オフライン統合（「夢」）**：記憶の昇格と減衰、新しい証拠による意味記憶の見直し、エピソードの出来事／話題ナラティブへのクラスタリング、コアサマリーとエンティティ投影の再構築、忘却候補のスキャン。
- **多人数の記憶**：話者帰属付きの共有履歴、ルーム単位のファクトカード、共有の collective スコープ、そして信頼度が減衰しながらエージェント間を伝わる「噂」。
- **キャストは設定で**：名前、別名、集団の呼び方、ユーザーの表示名、エージェントごとの文体ルールや語彙は、コードではなく設定です。

## インストール

```bash
pip install -e .                 # コア：numpy、scipy、jieba
pip install -e ".[llm]"          # OpenAI 互換クライアント（抽出 / 見直し / 返答）
pip install -e ".[embeddings]"   # 密ベクトル想起（sentence-transformers / FlagEmbedding、大きなダウンロード）
pip install -e ".[dev]"          # pytest、ruff、build
```

Python 3.11 以上が必要です。

## クイックスタート（オフライン、API キー不要）

```python
from tsuzuri import Tsuzuri

memory = Tsuzuri("./.tsuzuri")
memory.chat("林栖", "我下周三上午要去面试，有点紧张")
memory.chat("林栖", "对了，我不太能吃辣")

turn = memory.chat("林栖", "你还记得我下周三的面试吗？")
print(turn.reply)        # オフライン応答器がコンパイル済みワークスペースから答える
print(turn.recalled)     # このターンで想起された記憶
print(turn.fact_card)    # このやり取りから抽出されたファクトカード
```

> [!TIP]
> `tsuzuri demo` はスクリプト化された会話、想起チェック、グループルームを実行します。API キーもモデルのダウンロードも不要です。

LLM がない場合、事実抽出はルールベースになり、返答は決定的な代替（`tsuzuri.offline.offline_reply`）が生成します。記憶ループの動作確認、テスト、評価には十分です。実運用ではモデルを接続してください。

## モデルを使う

```python
from tsuzuri import Tsuzuri, openai_compatible

llm, chat = openai_compatible(model="your-model", base_url="https://...", api_key="...")
memory = Tsuzuri("./.tsuzuri", llm=llm, reply=chat)
```

`llm`（`prompt -> str`）は事実抽出・意味記憶の見直し・統合に、`reply`（`messages -> str`）はコンパイル済みメッセージからのキャラクターの返答生成に使います。記憶レイヤーとしてだけ使うこともできます：`memory.build_messages(agent, text)` の結果を自前のモデルスタックに渡してください。

## 独自のキャスト

[docs/configuration.md](docs/configuration.md) と `examples/custom_cast.json` を参照してください。`personas=`、`PERSONA_CONFIG_PATH`、または `configure_personas()` で読み込みます。

## グループ会話

```python
room = memory.room(["林栖", "苏芮", "叶澄"], shared=True)
turn = room.say("周六我们一起去爬山吧，七点楼下集合")
room.share("苏芮", "阿澈下个月要去参加摄影比赛")   # 他のメンバーが後で「聞いた話」として受け取る
```

## 📊 評価

`tsuzuri eval` は隔離された一時ストアで短い合成履歴を再生し、必要な情報が必要なときにモデルから見えるかを確認します。

| カテゴリ | オフライン・語彙想起 | + 密ベクトル（bge-small） |
|---|---|---|
| 好み | 1/3 | 1/3 |
| 今後の予定 | 0/3 | 1/3 |
| 安定した事実 | 1/2 | 2/2 |
| 訂正（最新が優先） | 1/2 | 1/2 |
| 隔離（漏れず、本人は想起できる） | 1/2 | 2/2 |
| グループルーム | 2/2 | 2/2 |
| 噂 | 1/1 | 1/1 |
| 複数ターンの流れ | 0/1 | 1/1 |
| **合計** | **7/16 (44%)** | **11/16 (69%)** |

同梱の合成セット v1。質問は**新しいセッション**で行います（原文もセッションのファクトカードもなく、長期記憶だけが答えられる）。抽出はルールベースで、LLM は使っていません。LLM 抽出は上表に含まれていません。再現：`tsuzuri eval`、`tsuzuri eval --embeddings`。ケース形式は [docs/evaluation.md](docs/evaluation.md) を参照してください。

> [!NOTE]
> オフラインの数値は語彙想起のみを測ります。質問と記憶に共通の語があれば想起されます（「面试」↔「面试」）が、意味だけが近い場合は想起されません（「我下周三要做什么」↔「面试」）。密ベクトルと LLM 抽出でこの差の多くは埋まります。ご自身の構成とケースで評価してください。

> [!WARNING]
> これは 16 ケースの**可視性**チェック（期待テキストがモデルの読む内容に含まれるか）であり、端から端までの回答正解率ではなく、LoCoMo / LongMemEval のスコアとは比較できません。ペルソナプロンプトは数えず、噂はコンパイル済みコンテキストに届いた場合のみ、隔離ケースは本人が想起できること（陽性対照）も条件です。

**既知の制限**：数日後の自然な追質問（「覚えてる？」なし）は表面化せず沈黙の根拠付けになりがちです。オフラインのルール抽出は発言の反復にすぎず、約束や未完了の話題は抽出せず、ルームの事実は長期記憶に保存されません。1 つのストアは 1 ユーザー専用で、ハード削除・エクスポート API はまだありません（データディレクトリを削除すれば消去できます）。

> [!IMPORTANT]
> ユーザーの発言はすべてデータディレクトリ（`memory.db`、`sessions/`）に保存される可能性があります。個人データとして保護し、API キーは環境変数かコミットしない `tsuzuri.toml` に置き、想起したテキストはモデルにとって信頼できない入力として扱ってください。

## アプリケーションとの統合

エンジンがアプリケーションを import することはありません。ホストしか知らない情報（キャラクターの完全なシステムプロンプト、設定上の事実、OOC 分類器、振り返りの配信先など）が必要な場面では、`tsuzuri.memory_host` を通じて任意の名前付きフックを要求します。フックがなければその手順はスキップされます。[docs/hosting.md](docs/hosting.md) を参照してください。

## コントリビューションとライセンス

Issue と Pull Request を歓迎します。まず [CONTRIBUTING.md](CONTRIBUTING.md) をお読みください。セキュリティ上の問題は [SECURITY.md](SECURITY.md) へ。Apache License 2.0 — [LICENSE](LICENSE) と [NOTICE](NOTICE) を参照してください。
