"""新闻标题情绪（论文第二部分（二）的第一至三步）。

第一、二步  每条标题经 FinBERT 得到三分类概率 V_{t,i} = [p_neg, p_neu, p_pos]（三项和为 1）
第三步      对某只股票第 t 个交易日的 N_t 条新闻做日度聚合：
            NS_t  = (Σ p_pos − Σ p_neg) / N_t                        式(1)，取值 [−1, 1]
            V̄_t  = (1/N_t) Σ V_{t,i}，d_i = ‖V_{t,i} − V̄_t‖₂，
            DIV_t = StDev(d_i)                                       式(2)
第四步（累积与衰减，式3、式4）在 features/dataset.py 中按交易日历完成。

FinBERTScorer : Hugging Face 中文金融情绪模型 finbert-tone-chinese（需安装 transformers）
LexiconScorer : 金融词典 + 否定词，把 [−1,1] 的词典得分近似映射为三分类概率，仅用于没有 GPU 时调试流程
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

POS_WORDS = """增长 大增 预增 扭亏 盈利 超预期 新高 创新高 突破 中标 订单 签约 签订 合作 战略合作 获批 批准 通过 上市
回购 增持 分红 派息 高送转 涨停 大涨 上涨 走强 领涨 利好 受益 景气 提升 加速 扩产 投产 量产 首发 发布 推出 研发成功
入选 获奖 龙头 看好 买入 推荐 上调 增资 注资 并购 收购 重组成功 解禁结束 摘帽 业绩亮眼 翻倍 同比增长 净利增长""".split()
NEG_WORDS = """下滑 下降 下跌 大跌 跌停 走弱 亏损 预亏 预减 首亏 续亏 减持 清仓 质押 冻结 立案 调查 处罚 罚款 警示 问询
留置 被查 违规 违法 造假 诉讼 仲裁 退市 风险警示 戴帽 爆雷 违约 逾期 债务 终止 取消 失败 暂停 停产 召回 事故 下调
卖出 利空 承压 萎缩 流出 跳水 闪崩 腰斩 商誉减值 减值 计提 解禁 辞职 失联 被执行""".split()
NEGATIONS = ['未', '不', '无', '没有', '否认', '并非', '免于', '解除']
PROB_COLS = ['p_neg', 'p_neu', 'p_pos']


class LexiconScorer:
    """词典得分 s ∈ (−1,1) 近似为概率：p_pos = max(s,0)，p_neg = max(−s,0)，p_neu = 1 − |s|。"""

    def __init__(self, extra_pos=None, extra_neg=None):
        pos, neg = set(POS_WORDS), set(NEG_WORDS)
        for path, s in [(extra_pos, pos), (extra_neg, neg)]:
            if path:
                s.update(w.strip() for w in Path(path).read_text(encoding='utf-8').splitlines() if w.strip())
        self.words = sorted([(w, 1) for w in pos] + [(w, -1) for w in neg], key=lambda x: -len(x[0]))

    def score_one(self, text: str) -> float:
        s, used = 0.0, [False] * len(text)
        for w, sign in self.words:
            start = text.find(w)
            while start != -1:
                if not any(used[start:start + len(w)]):
                    for i in range(start, start + len(w)):
                        used[i] = True
                    pre = text[max(0, start - 3):start]
                    s += -sign if any(n in pre for n in NEGATIONS) else sign
                start = text.find(w, start + 1)
        return float(np.tanh(s / 2))

    def score(self, texts):
        s = np.array([self.score_one(t) for t in tqdm(texts, desc='词典情绪打分', mininterval=5)])
        return np.stack([np.clip(-s, 0, None), 1 - np.abs(s), np.clip(s, 0, None)], axis=1)


class FinBERTScorer:
    """论文第一、二步：分词（加 [CLS]/[SEP]，最大长度 256），取 [CLS] 输出经三分类层与 Softmax 得到概率。"""

    def __init__(self, model_name, batch_size=64, max_length=256, label_map=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.dev)
        self.bs, self.max_len = batch_size, max_length
        id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.order = self._label_order(id2label, label_map or {})

    @staticmethod
    def _label_order(id2label, label_map):
        """返回 [neg, neu, pos] 对应的 logits 下标。"""
        kind = {}
        for i, name in id2label.items():
            k = label_map.get(name, name).lower()
            if any(x in k for x in ['neg', '负', 'bear']):
                kind['neg'] = i
            elif any(x in k for x in ['neu', '中']):
                kind['neu'] = i
            elif any(x in k for x in ['pos', '正', 'bull']):
                kind['pos'] = i
        if len(kind) != 3:
            raise ValueError(f'无法识别模型标签 {id2label}，请在 sentiment.label_map 中指定，'
                             '如 {LABEL_0: neu, LABEL_1: pos, LABEL_2: neg}')
        return [kind['neg'], kind['neu'], kind['pos']]

    def score(self, texts):
        out = []
        with self.torch.no_grad():
            for i in tqdm(range(0, len(texts), self.bs), desc='FinBERT 情绪概率'):
                enc = self.tok(list(texts[i:i + self.bs]), padding=True, truncation=True,
                               max_length=self.max_len, return_tensors='pt').to(self.dev)
                p = self.torch.softmax(self.model(**enc).logits, -1)[:, self.order]
                out.append(p.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3))


def get_scorer(cfg):
    c = cfg['sentiment']
    if c['method'] in ('finbert', 'transformer'):
        return FinBERTScorer(c['model_name'], c['batch_size'], c.get('max_length', 256), c.get('label_map'))
    return LexiconScorer(c.get('extra_pos_dict'), c.get('extra_neg_dict'))


def daily_sentiment(news: pd.DataFrame):
    """第三步，个股日度聚合：NS（式1）、DIV（式2）与新闻条数 N_t。news 需含 p_neg/p_neu/p_pos 三列。"""
    keys = ['order_book_id', 'date']
    df = news[keys + PROB_COLS].copy()
    mean = df.groupby(keys)[PROB_COLS].transform('mean')
    df['d'] = np.sqrt(((df[PROB_COLS].to_numpy() - mean.to_numpy()) ** 2).sum(1))   # d_i = ‖V_{t,i} − V̄_t‖₂
    df['net'] = df['p_pos'] - df['p_neg']
    g = df.groupby(keys)
    out = pd.DataFrame({'ns': g['net'].mean(),                 # (Σp_pos − Σp_neg)/N_t
                        'div': g['d'].std(ddof=0),               # StDev(d_i)，单条新闻时为 0
                        'news_cnt': g.size()}).reset_index()
    return out
