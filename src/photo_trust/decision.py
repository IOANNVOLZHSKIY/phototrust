"""Decision layer: turns model signals + listing context into one moderation action.

The model never bans anyone. Outcomes:
  PUBLISH            - published instantly (the 'fast lane')
  SUGGEST_AI_LABEL   - published after the seller confirms an "Illustration (AI)" label
                       or replaces the photo; seller can dispute -> moderator
  REVIEW             - goes to the moderator queue with an explanation
  HIDE_AND_REVIEW    - hidden from search until a moderator decides (very confident copy)
"""
from dataclasses import dataclass, field


@dataclass
class Thresholds:
    copy_review: float          # similarity at which a match goes to moderator (FPR <= 1% on honest)
    copy_hide: float            # near-certain copy of another seller's photo
    ai_label: float             # P(AI) above which we suggest the AI label
    stock_min_listings: int = 3  # the same photo in >= N listings of different sellers = stock/box photo


@dataclass
class Match:
    listing_id: str
    seller_id: str
    similarity: float
    n_listings_with_photo: int = 1   # how many listings in the base use (nearly) this photo


@dataclass
class Decision:
    action: str
    reasons: list = field(default_factory=list)


def decide(seller_id, top_match, ai_score, th: Thresholds, price_ratio=None):
    """price_ratio = listing price / median price of similar listings (None if unknown)."""
    reasons = []
    if top_match is not None and top_match.similarity >= th.copy_review:
        same_seller = top_match.seller_id == seller_id
        stock = top_match.n_listings_with_photo >= th.stock_min_listings
        if same_seller:
            return Decision("REVIEW", [f"Дубль своего объявления {top_match.listing_id} "
                                       f"(сходство {top_match.similarity:.2f}) → склейка, не санкция"])
        if stock:
            reasons.append(f"Фото встречается в {top_match.n_listings_with_photo} объявлениях — похоже на "
                           "стоковое фото / фото коробки, это не нарушение")
        else:
            cheap = price_ratio is not None and price_ratio < 0.8
            reasons.append(f"Совпадает с фото другого продавца в {top_match.listing_id} "
                           f"(сходство {top_match.similarity:.2f})")
            if cheap:
                reasons.append(f"Цена ниже рынка на {100 * (1 - price_ratio):.0f}%")
            # Hiding is reserved for near-certain copies; a low price only adds context for the moderator.
            action = "HIDE_AND_REVIEW" if top_match.similarity >= th.copy_hide else "REVIEW"
            return Decision(action, reasons)
    if ai_score is not None and ai_score >= th.ai_label:
        return Decision("SUGGEST_AI_LABEL", reasons + [f"Фото похоже на сгенерированное ИИ (P={ai_score:.2f})"])
    return Decision("PUBLISH", reasons or ["Сигналов нет"])


ACTION_RU = {
    "PUBLISH": "Опубликовать сразу",
    "SUGGEST_AI_LABEL": "Предложить метку «Иллюстрация (ИИ)» / замену фото",
    "REVIEW": "Отправить модератору",
    "HIDE_AND_REVIEW": "Скрыть из выдачи и отправить модератору",
}
