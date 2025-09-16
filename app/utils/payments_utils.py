# --- [NEW] Payments SDK imports ---------------------------------------------
import json
import os
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Dict, Any, Optional, Iterable, List

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

# Se hai salvato lo SDK in app/payments_sdk/teatek_me_plans_sdk.py:
from app.payments_sdk.sdk import (
    MePlansClient,
    DynamicCheckoutRequest,
    PortalSessionRequest,
    PortalConfigSelector,
    PortalUpdateDeepLinkRequest,
    PortalCancelDeepLinkRequest,
    ResourcesState,
    DynamicResource, ConsumeResourcesRequest, ResourceItem, ApiError,
)


########################################################################################################################
# --- [NEW] Payments config ---------------------------------------------------
PLANS_API_BASE         = os.getenv("PLANS_API_BASE", "http://localhost:8222").rstrip("/")
PLANS_ADMIN_API_KEY    = os.getenv("PLANS_ADMIN_API_KEY", "adminkey123:admin")
PLANS_STRIPE_ACCOUNT   = os.getenv("PLANS_STRIPE_ACCOUNT", None)
PLANS_SUCCESS_URL_DEF  = os.getenv("PLANS_SUCCESS_URL", "https://tuo-sito.com/success?cs_id={CHECKOUT_SESSION_ID}")
PLANS_CANCEL_URL_DEF   = os.getenv("PLANS_CANCEL_URL",  "https://tuo-sito.com/cancel")
RETURN_URL = None
# Stati subscription considerate “vive”
ALIVE_SUB_STATUSES = {"trialing","active","past_due","unpaid","incomplete","paused","incomplete_expired"}

# Bucket varianti di catalogo che vuoi mostrare nel Portal/Deeplink (puoi adattarlo)
# 👇 Catalogo ufficiale 6 varianti (label e prezzo in centesimi: EUR)
VARIANTS_CATALOG: Dict[str, Dict[str, Any]] = {
    # mensili
    "starter_monthly":    {"label": "Starter (Mensile)",    "unit_amount": 199,  "period": "monthly"},
    "premium_monthly":    {"label": "Premium (Mensile)",    "unit_amount": 499,  "period": "monthly"},
    "enterprise_monthly": {"label": "Enterprise (Mensile)", "unit_amount": 999,  "period": "monthly"},
    # annuali
    "starter_annual":     {"label": "Starter (Annuale)",    "unit_amount": 1990, "period": "annual"},
    "premium_annual":     {"label": "Premium (Annuale)",    "unit_amount": 4990, "period": "annual"},
    "enterprise_annual":  {"label": "Enterprise (Annuale)", "unit_amount": 9990, "period": "annual"},
}

# Bucket completo per Portal/Deeplink (ordine di default)
VARIANTS_BUCKET = [
    "starter_monthly", "premium_monthly", "enterprise_monthly",
    "starter_annual",  "premium_annual",  "enterprise_annual",
]

try:
    PLAN_VALUE_MAP: Dict[str, int] = json.loads(os.getenv("PLANS_VARIANT_VALUES", "{}")) or {}
except Exception:
    PLAN_VALUE_MAP = {}

# Se non fornito da env, imposta un ranking sensato (starter<premium<enterprise, mensile<annuale)
DEFAULT_VALUES = {
    "starter_monthly": 1, "premium_monthly": 2, "enterprise_monthly": 3,
    "starter_annual":  4, "premium_annual":  5, "enterprise_annual":  6,
}
for k, v in DEFAULT_VALUES.items():
    PLAN_VALUE_MAP.setdefault(k, v)
########################################################################################################################

# --- [NEW] Helpers SDK Payments ---------------------------------------------
def _mk_plans_client(access_token: str) -> MePlansClient:
    """
    Crea il client dello SDK MePlans usando:
      - JWT dell’utente (Bearer)
      - Admin API Key per endpoint privilegiati (Portal/Deeplink/resources)
    """
    if not access_token:
        raise HTTPException(401, "Access token mancante")
    return MePlansClient(
        api_base=PLANS_API_BASE,
        access_token=access_token,
        admin_api_key=PLANS_ADMIN_API_KEY,
        stripe_account=PLANS_STRIPE_ACCOUNT,
        default_timeout=40.0,
    )

async def _sdk(callable_, *args, **kwargs):
    """
    Esegue un metodo sincrono dello SDK in thread-pool per non bloccare l’event loop FastAPI.
    """
    return await run_in_threadpool(lambda: callable_(*args, **kwargs))

async def _find_current_subscription_id(client: MePlansClient) -> str | None:
    """
    Restituisce la subscription 'viva' più recente dell’utente (o None se non trovata).
    """
    lo = await _sdk(client.list_subscriptions, limit=10)
    data = lo.data if hasattr(lo, "data") else (lo or {}).get("data") or []
    if not data:
        return None
    # prendi la più recente fra gli stati 'alive'
    data_sorted = sorted(data, key=lambda s: int(s.get("created") or 0), reverse=True)
    for s in data_sorted:
        st = (s.get("status") or "").lower()
        if st in ALIVE_SUB_STATUSES:
            return s.get("id")
    return None

def _variant_to_portal_preset(variant: str | None) -> str:
    return "annual" if (variant or "").endswith("_annual") else "monthly"

def _dataclass_to_dict(x):
    return asdict(x) if is_dataclass(x) else x

def _variant_value(variant: str) -> int:
    if variant in PLAN_VALUE_MAP:
        return PLAN_VALUE_MAP[variant]
    try:
        return 1 + VARIANTS_BUCKET.index(variant)
    except ValueError:
        return 10_000  # sconosciuti in coda

def _sorted_variants(variants: Iterable[str]) -> List[str]:
    return sorted({v for v in variants}, key=_variant_value)

class ChangeIntent(str, Enum):
    upgrade   = "upgrade"
    downgrade = "downgrade"
    both      = "both"
    none      = "none"

def build_variants_for_intent(
    *,
    current_variant: Optional[str],
    intent: Optional[ChangeIntent],
    catalog: Optional[Iterable[str]] = None
) -> List[str]:
    """
    - upgrade:    solo piani con valore > del corrente + il corrente
    - downgrade:  solo piani con valore < del corrente + il corrente
    - both/None:  tutto il catalogo
    """
    bucket = list(catalog or VARIANTS_BUCKET)
    intent = intent or ChangeIntent.both

    if not current_variant or current_variant not in bucket:
        return _sorted_variants(bucket)

    cur_val = _variant_value(current_variant)
    if intent == ChangeIntent.upgrade:
        out = [v for v in bucket if _variant_value(v) > cur_val] + [current_variant]
    elif intent == ChangeIntent.downgrade:
        out = [v for v in bucket if _variant_value(v) < cur_val] + [current_variant]
    else:
        out = bucket

    return _sorted_variants(out)

def features_for_update(intent: Optional[ChangeIntent]) -> Dict[str, Any]:
    base = {"payment_method_update": {"enabled": True}}

    if intent == ChangeIntent.upgrade:
        # Cambio piano consentito e immediato, differenza pagata subito
        return {
            **base,
            "subscription_update": {
                "enabled": True,
                "default_allowed_updates": ["price"],
                "proration_behavior": "always_invoice",
            },
            "subscription_cancel": {"enabled": True, "mode": "at_period_end"},
        }

    elif intent == ChangeIntent.downgrade:
        # Il Portal NON fa update; il downgrade lo programmi dal server (schedule)
        return {
            **base,
            "subscription_update": {
                "enabled": True,
                "default_allowed_updates": ["price"],
                "proration_behavior": "none",
            },
            "subscription_cancel": {"enabled": True, "mode": "at_period_end"},
        }

    elif intent == ChangeIntent.both:
        # Compromesso unico per entrambe le direzioni: prorate ora, conguaglio alla prossima fattura
        return {
            **base,
            "subscription_update": {
                "enabled": True,
                "default_allowed_updates": ["price"],
                "proration_behavior": "create_prorations",
            },
            "subscription_cancel": {"enabled": True, "mode": "at_period_end"},
        }

    elif intent == ChangeIntent.none:
        # Nessun cambio piano via Portal
        return {
            **base,
            "subscription_update": {"enabled": False},
            "subscription_cancel": {"enabled": True, "mode": "at_period_end"},
        }




# --- [NEW] Caches L2 (in-memory, per-process) --------------------------------
import time, hashlib

_CONFIG_TTL_SEC = int(os.getenv("PLANS_PORTAL_CONFIG_TTL_SEC", "3600"))   # 1h
_PRICE_TTL_SEC  = int(os.getenv("PLANS_PRICE_ID_TTL_SEC", "86400"))      # 24h

_config_cache: Dict[str, tuple[str, float]] = {}   # key -> (configuration_id, expires_at)
_price_cache:  Dict[tuple[str, str], tuple[str, float]] = {}  # (plan_type, variant) -> (price_id, expires_at)

def _cfg_key(plan_type: str, variants: List[str], features: Dict[str, Any], headline: str) -> str:
    payload = {"plan_type": plan_type, "variants": variants, "features": features, "headline": headline}
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _config_cache_get(key: str) -> Optional[str]:
    v = _config_cache.get(key)
    if not v: return None
    cid, exp = v
    if time.time() > exp:
        _config_cache.pop(key, None)
        return None
    return cid

def _config_cache_put(key: str, configuration_id: str) -> None:
    _config_cache[key] = (configuration_id, time.time() + _CONFIG_TTL_SEC)

def _price_cache_get(plan_type: str, variant: str) -> Optional[str]:
    v = _price_cache.get((plan_type, variant))
    if not v: return None
    pid, exp = v
    if time.time() > exp:
        _price_cache.pop((plan_type, variant), None)
        return None
    return pid

def _price_cache_put(plan_type: str, variant: str, price_id: str) -> None:
    _price_cache[(plan_type, variant)] = (price_id, time.time() + _PRICE_TTL_SEC)


# ============================================================
# C R E D I T S :  consumo prima dell’operazione
# ============================================================
import uuid

async def _consume_credits_or_402(
    token: str | None,
    amount: float,
    *,
    reason: str,
    expected_plan_type: str | None = None,
    expected_variant: str | None = None,
    subscription_id: str | None = None
) -> None:
    """
    Consuma 'amount' crediti dall’utente; se fallisce -> HTTP 402/4xx.
    NOTA: 'amount' è già in crediti (nessuna conversione USD->crediti).
    """
    if not token:
        raise HTTPException(401, "Token richiesto per consumo crediti")

    if amount is None or amount <= 0:
        return  # nulla da consumare

    t_1 = time.time()
    client = _mk_plans_client(token)
    t_2 = time.time()
    sub_id = subscription_id or await _find_current_subscription_id(client)
    t_3 = time.time()
    if not sub_id:
        raise HTTPException(404, "Nessuna subscription attiva trovata")

    # Request id per debugging/idempotency lato log
    req_id = str(uuid.uuid4())
    body = ConsumeResourcesRequest(
        items=[ResourceItem(key="credits", unit="credits", quantity=round(float(amount),0))],
        reason=f"{reason} | req_id={req_id}",
        expected_plan_type=expected_plan_type,
        expected_variant=expected_variant,
    )
    try:
        # Lo SDK del client è sincrono: usa wrapper _sdk (già presente)
        t_4 = time.time()
        await _sdk(client.consume_resources, sub_id, body)
        t_5 = time.time()
        print("#*" * 120)
        print(t_2 - t_1, t_3 - t_2, t_4 - t_3, t_5 - t_4)
        print("#*" * 120)
    except ApiError as e:
        # Propaga il payload originale del backend piani
        raise HTTPException(
            status_code=getattr(e, "status_code", 402),
            detail=getattr(e, "payload", str(e))
        )
