#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo end-to-end per API PAGAMENTI — LIVELLO 2 (6 piani)
Flusso:
  0) Login (Cognito) → AccessToken
  1) /payments/current_plan
     - se 404 → /payments/checkout (gestisce anche 409 portal_redirect)
     - in parallelo /payments/portal_session
  2) Attende la subscription (poll su /payments/current_plan) o INVIO manuale
  3) /payments/credits
  4) Deeplink:
       - /payments/deeplink/update (intent both, catalogo completo 6 varianti)
       - /payments/deeplink/upgrade (adiacente ↑ e ↓ nel bucket)
       - /payments/deeplink/cancel (immediate)
  5) (OPZ) consumo reale crediti via L1 (disabilitato di default)

Requisiti:
  pip install requests python-dotenv
"""

from __future__ import annotations
import os
import sys
import time
import json
from typing import Dict, Any, Optional, Tuple, List

import requests
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

# Base URL livello 2 (fallback alla vecchia variabile per retro-compatibilità)
L2_API_BASE         = os.getenv("PLANS_L2_API_BASE", os.getenv("PLANS_API_BASE", "http://localhost:8888")).rstrip("/")
AUTH_API_BASE       = os.getenv("AUTH_API_BASE", "https://teatek-llm.theia-innovation.com/auth").rstrip("/")
RETURN_URL_DEFAULT  = os.getenv("RETURN_URL", "https://tuo-sito.com/account")

# Credenziali demo (solo test/manuale)
USERNAME: str = os.getenv("DEMO_USERNAME", "sansalonesimone0@gmail.com")
PASSWORD: str = os.getenv("DEMO_PASSWORD", "h326JH%gesL")

# Pianificazione/variant
PLAN_TYPE_DEFAULT   = os.getenv("PLAN_TYPE_DEFAULT", "ai_standard").strip()   # ⬅️ nuovo plan_type
START_VARIANT       = os.getenv("START_VARIANT", "starter_monthly").strip()   # ⬅️ default: Starter mensile

# Bucket completo (ordinato) con le 6 varianti
VARIANTS_BUCKET = [
    v.strip() for v in os.getenv(
        "VARIANTS_BUCKET",
        "starter_monthly,premium_monthly,enterprise_monthly,starter_annual,premium_annual,enterprise_annual"
    ).split(",") if v.strip()
]

# Opzione: consumo *reale* crediti (via L1) – DISABILITATO di default
ENABLE_REAL_CONSUME = (os.getenv("ENABLE_REAL_CONSUME", "false").lower() == "true")
L1_API_BASE         = os.getenv("PLANS_L1_API_BASE", "http://localhost:8001").rstrip("/")
L1_ADMIN_API_KEY    = os.getenv("PLANS_L1_ADMIN_API_KEY")  # richiesto solo se ENABLE_REAL_CONSUME=true
L1_STRIPE_ACCOUNT   = os.getenv("PLANS_L1_STRIPE_ACCOUNT") or None

# Per comodità: chiedere INVIO dopo checkout
WAIT_AFTER_CHECKOUT = True

# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────
def _u(path: str) -> str:
    return f"{L2_API_BASE}{path}"

def _req(method: str, url: str, *, params: Dict[str, Any] | None = None, json_body: Dict[str, Any] | None = None, timeout: float = 40.0) -> Any:
    r = requests.request(method, url, params=params, json=json_body, timeout=timeout)
    ct = (r.headers.get("content-type") or "")
    if r.status_code >= 300:
        try:
            payload = r.json() if "application/json" in ct else r.text
        except Exception:
            payload = r.text
        raise RuntimeError(f"HTTP {r.status_code} {url}: {payload}")
    if not r.content:
        return None
    try:
        return r.json() if "application/json" in ct else r.text
    except Exception:
        return r.text

# ─────────────────────────────────────────────────────────────────────────────
# Auth (Cognito minimal)
# ─────────────────────────────────────────────────────────────────────────────
def signin_and_get_access_token() -> str:
    url = f"{AUTH_API_BASE}/v1/user/signin"
    body = {"username": USERNAME, "password": PASSWORD}
    r = requests.post(url, json=body, timeout=40)
    r.raise_for_status()
    data = r.json()
    token = (data.get("AuthenticationResult") or {}).get("AccessToken")
    if not token:
        raise RuntimeError(f"Signin OK ma AccessToken non trovato: {data}")
    return token

# ─────────────────────────────────────────────────────────────────────────────
# API livello 2 — chiamate
# ─────────────────────────────────────────────────────────────────────────────
def api_current_plan(token: str) -> Dict[str, Any]:
    return _req("GET", _u("/payments/current_plan"), params={"token": token})

def api_credits(token: str, subscription_id: Optional[str] = None) -> Dict[str, Any]:
    params = {"token": token}
    if subscription_id:
        params["subscription_id"] = subscription_id
    return _req("GET", _u("/payments/credits"), params=params)

def api_checkout_variant(
    token: str,
    *,
    plan_type: str,
    variant: str,
    success_url: Optional[str] = None,
    cancel_url: Optional[str] = None,
    locale: str = "it"
) -> Dict[str, Any]:
    body = {
        "token": token,
        "plan_type": plan_type,
        "variant": variant,
        "locale": locale,
        "success_url": success_url or f"{RETURN_URL_DEFAULT}?cs_id={{CHECKOUT_SESSION_ID}}",
        "cancel_url":  cancel_url  or "https://tuo-sito.com/cancel",
    }
    return _req("POST", _u("/payments/checkout"), json_body=body)

def api_portal_session(token: str, return_url: Optional[str] = None) -> Dict[str, Any]:
    body = {"token": token, "return_url": (return_url or RETURN_URL_DEFAULT)}
    return _req("POST", _u("/payments/portal_session"), json_body=body)

def api_deeplink_update(
    token: str,
    *,
    return_url: Optional[str] = None,
    change_intent: str = "both",
    variants_override: Optional[List[str]] = None,
    variants_catalog: Optional[List[str]] = None
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "token": token,
        "return_url": (return_url or RETURN_URL_DEFAULT),
        "change_intent": change_intent,
    }
    if variants_override:
        body["variants_override"] = variants_override
    if variants_catalog:
        body["variants_catalog"] = variants_catalog
    return _req("POST", _u("/payments/deeplink/update"), json_body=body)

def api_deeplink_upgrade(
    token: str,
    *,
    return_url: Optional[str] = None,
    target_plan_type: Optional[str] = None,
    target_variant: Optional[str] = None,
    target_price_id: Optional[str] = None,
    quantity: int = 1
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "token": token,
        "return_url": (return_url or RETURN_URL_DEFAULT),
        "quantity": quantity,
    }
    if target_price_id:
        body["target_price_id"] = target_price_id
    else:
        body["target_plan_type"] = target_plan_type
        body["target_variant"]   = target_variant
    return _req("POST", _u("/payments/deeplink/upgrade"), json_body=body)

def api_deeplink_cancel(
    token: str,
    *,
    return_url: Optional[str] = None,
    immediate: bool = True,
    variants_catalog: Optional[List[str]] = None,
    portal_preset: Optional[str] = None
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "token": token,
        "return_url": (return_url or RETURN_URL_DEFAULT),
        "immediate": immediate,
    }
    if variants_catalog:
        body["variants_catalog"] = variants_catalog
    if portal_preset:
        body["portal_preset"] = portal_preset
    return _req("POST", _u("/payments/deeplink/cancel"), json_body=body)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers demo
# ─────────────────────────────────────────────────────────────────────────────
def _adjacent_variants(current: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if current in VARIANTS_BUCKET:
        i = VARIANTS_BUCKET.index(current)
    else:
        i = 0
    down_v = VARIANTS_BUCKET[i - 1] if i - 1 >= 0 else None
    up_v   = VARIANTS_BUCKET[i + 1] if i + 1 < len(VARIANTS_BUCKET) else None
    return down_v, up_v

def wait_for_current_plan(token: str, *, timeout_sec: int = 180, poll_every: float = 3.0) -> Optional[Dict[str, Any]]:
    print(f"[Wait] attendo comparsa subscription (max {timeout_sec}s)…")
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            cp = api_current_plan(token)
            if cp and cp.get("subscription_id"):
                return cp
        except Exception:
            pass
        time.sleep(poll_every)
    return None

def print_current_plan(cp: Dict[str, Any]) -> None:
    print("\n[Current Plan]")
    print(f"  Subscription: {cp.get('subscription_id')}")
    print(f"  Status:       {cp.get('status')}")
    print(f"  Plan Type:    {cp.get('plan_type')}")
    print(f"  Variant:      {cp.get('variant')}")
    print(f"  Price:        {cp.get('active_price_id')}")
    ps = cp.get("period_start"); pe = cp.get("period_end")
    if ps and pe:
        from datetime import datetime, timezone
        ps_dt = datetime.fromtimestamp(int(ps), tz=timezone.utc)
        pe_dt = datetime.fromtimestamp(int(pe), tz=timezone.utc)
        print(f"  Period:       {ps_dt.isoformat()} → {pe_dt.isoformat()}")

def print_credits(label: str, c: Dict[str, Any]) -> None:
    print(f"\n[{label} — Credits]")
    print("  Provided: ", c.get("provided_total"))
    print("  Used:     ", c.get("used_total"))
    print("  Remaining:", c.get("remaining_total"))

# ─────────────────────────────────────────────────────────────────────────────
# (OPZIONALE) Consumo reale via L1 (solo se esplicitamente abilitato)
# ─────────────────────────────────────────────────────────────────────────────
def _consume_via_level1(token: str, subscription_id: str, quantity: int = 5) -> None:
    if not ENABLE_REAL_CONSUME:
        print("\n[Consume] Skippato (ENABLE_REAL_CONSUME=false).")
        return
    if not L1_ADMIN_API_KEY:
        print("\n[Consume][WARN] PLANS_L1_ADMIN_API_KEY mancante — impossibile consumare davvero.")
        return
    url = f"{L1_API_BASE}/me/plans/subscriptions/{subscription_id}/resources/consume"
    body = {
        "items": [{"key": "credits", "quantity": float(quantity), "unit": "credits"}],
        "reason": "demo_consume_level2",
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "X-API-Key": L1_ADMIN_API_KEY,
        "Content-Type": "application/json",
    }
    if L1_STRIPE_ACCOUNT:
        headers["x-stripe-account"] = L1_STRIPE_ACCOUNT
    print(f"\n[Consume] Provo consumo REALE {quantity} crediti (via L1: {url}) …")
    r = requests.post(url, json=body, headers=headers, timeout=40)
    if r.status_code >= 300:
        try:
            print("[Consume][ERR] HTTP", r.status_code, r.json())
        except Exception:
            print("[Consume][ERR] HTTP", r.status_code, r.text)
        return
    try:
        print("[Consume] OK:", r.json().get("resources", {}))
    except Exception:
        print("[Consume] OK")

# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("[Auth] Login…")
    token = signin_and_get_access_token()
    print("[Auth] Access token ottenuto.")

    # 1) Current plan (se esiste)
    existing = None
    try:
        existing = api_current_plan(token)
        print_current_plan(existing)
    except Exception as e:
        print("[Info] Nessun piano attivo (o errore benigno):", e)

    # 2) Se NON esiste, avvia checkout (es. starter_monthly su ai_standard)
    if not (existing and existing.get("subscription_id")):
        print(f"\n[Checkout] Creo sessione per variant '{START_VARIANT}' del plan '{PLAN_TYPE_DEFAULT}' …")
        try:
            co = api_checkout_variant(token, plan_type=PLAN_TYPE_DEFAULT, variant=START_VARIANT)
            status = co.get("status")
            if status == "checkout":
                print("  Checkout Session ID:", co.get("checkout_session_id"))
                print("  URL:                 ", co.get("url"))
                # Portal session "in parallelo" (solo come comodità)
                try:
                    portal = api_portal_session(token, RETURN_URL_DEFAULT)
                    print("  Portal URL:          ", portal.get("url"))
                except Exception as e:
                    print("[Portal][WARN]", e)
                if WAIT_AFTER_CHECKOUT:
                    input("\n>>> Apri l’URL di checkout nel browser, completa il pagamento e poi premi INVIO… ")
                # Attendo che compaia la subscription
                cp = wait_for_current_plan(token, timeout_sec=180, poll_every=3.0)
                if not cp:
                    print("[WARN] Subscription non rilevata entro il timeout. Proseguo comunque.")
                else:
                    existing = cp
                    print_current_plan(existing)
            elif status == "portal_redirect":
                print("  Esiste già una subscription viva → redirect al Billing Portal:")
                print("  Portal URL:", co.get("portal_url"))
                # provo a leggere comunque il piano
                try:
                    existing = api_current_plan(token)
                    print_current_plan(existing)
                except Exception as e:
                    print("[Info] current_plan non disponibile subito:", e)
            else:
                print("[WARN] Risposta inattesa /payments/checkout:", co)
        except Exception as e:
            print("[ERROR] Checkout:", e)
    else:
        # Portal session comunque disponibile
        try:
            portal = api_portal_session(token, RETURN_URL_DEFAULT)
            print("\n[Portal] URL:", portal.get("url"))
        except Exception as e:
            print("[Portal][WARN]", e)

    # 3) Se ho una subscription, continuo con deeplink+crediti
    if not (existing and existing.get("subscription_id")):
        print("\n[Stop] Non ho una subscription attiva. Fine demo.")
        return

    sub_id   = existing["subscription_id"]
    cur_plan = existing.get("plan_type") or PLAN_TYPE_DEFAULT
    cur_var  = existing.get("variant")

    # Crediti (prima)
    try:
        credits_before = api_credits(token, sub_id)
        print_credits("PRIMA", credits_before)
    except Exception as e:
        print("[Credits][WARN]", e)

    # 4) Deeplink UPDATE (intent both, catalogo completo 6 varianti)
    try:
        dl_upd = api_deeplink_update(
            token,
            change_intent="both",
            variants_catalog=VARIANTS_BUCKET,  # ⬅️ passa le 6 varianti
            return_url=RETURN_URL_DEFAULT
        )
        print("\n[Deeplink][UPDATE] URL:", dl_upd.get("url"))
    except Exception as e:
        print("[Deeplink][UPDATE][WARN]", e)

    # 5) Deeplink UPGRADE e DOWNGRADE rispetto all’adiacenza nel bucket
    down_v, up_v = _adjacent_variants(cur_var)

    if up_v:
        try:
            dl_up = api_deeplink_upgrade(
                token,
                target_plan_type=cur_plan,
                target_variant=up_v,
                return_url=RETURN_URL_DEFAULT,
                quantity=1,
            )
            print("[Deeplink][UPGRADE] (→", up_v, ") URL:", dl_up.get("url"))
        except Exception as e:
            print("[Deeplink][UPGRADE][WARN]", e)
    else:
        print("[Deeplink][UPGRADE] Nessuna variante adiacente superiore nel bucket.")

    if down_v:
        try:
            dl_down = api_deeplink_upgrade(
                token,
                target_plan_type=cur_plan,
                target_variant=down_v,
                return_url=RETURN_URL_DEFAULT,
                quantity=1,
            )
            print("[Deeplink][DOWNGRADE] (→", down_v, ") URL:", dl_down.get("url"))
        except Exception as e:
            print("[Deeplink][DOWNGRADE][WARN]", e)
    else:
        print("[Deeplink][DOWNGRADE] Nessuna variante adiacente inferiore nel bucket.")

    # 6) Deeplink CANCEL (immediate)
    try:
        dl_can = api_deeplink_cancel(
            token,
            return_url=RETURN_URL_DEFAULT,
            immediate=True,
            variants_catalog=VARIANTS_BUCKET
        )
        print("[Deeplink][CANCEL][immediate] URL:", dl_can.get("url"))
    except Exception as e:
        print("[Deeplink][CANCEL][WARN]", e)

    # 7) (OPZ) Consumo REALE (via L1) + lettura crediti dopo
    try:
        _consume_via_level1(token, sub_id, quantity=5)
    except Exception as e:
        print("[Consume][WARN]", e)

    # Crediti (dopo)
    try:
        credits_after = api_credits(token, sub_id)
        print_credits("DOPO", credits_after)
    except Exception as e:
        print("[Credits][WARN]", e)

    print("\n[Done] Demo livello 2 completata.")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto dall’utente.")
        sys.exit(130)
