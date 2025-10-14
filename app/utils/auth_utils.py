# app/utils/auth_utils.py
from typing import Any, Dict, Optional
from fastapi import HTTPException
from app.auth_sdk.sdk import CognitoSDK, AccessTokenRequest

def fetch_user_info_by_token(sdk: CognitoSDK, access_token: str) -> Dict[str, Any]:
    """
    Chiama l'SDK Cognito per ottenere le informazioni complete dell'utente
    a partire dall'access token (mapping su /v1/user/user-info).

    Ritorna il dict con la stessa struttura dell'esempio Swagger.
    Lancia HTTPException 401/500 in caso di problemi.
    """
    if not access_token:
        raise HTTPException(status_code=401, detail="Access token mancante")

    try:
        data = AccessTokenRequest(access_token=access_token)
        return sdk.get_user_info(data)  # ← chiama l'endpoint /v1/user/user-info
    except Exception as e:
        # .raise_for_status() nello SDK rilancia già per HTTP != 2xx
        raise HTTPException(status_code=401, detail=f"Impossibile recuperare user-info: {e}")

def extract_username_from_userinfo(userinfo: Dict[str, Any]) -> str:
    """
    Estrae lo username dal payload di user-info.
    Ordine di preferenza:
      1) campo root 'Username'
      2) attributo 'preferred_username' (se presente)
      3) attributo 'email' (fallback “parlante”)
      4) attributo 'sub' (ultimo fallback)
    """
    if not isinstance(userinfo, dict):
        raise HTTPException(status_code=500, detail="User-info non valido")

    # 1) Campo root (caso standard, vedi esempio)
    root_username = userinfo.get("Username")
    if isinstance(root_username, str) and root_username.strip():
        return root_username

    # 2) Cerca tra gli attributi
    attrs = {a.get("Name"): a.get("Value") for a in userinfo.get("UserAttributes", []) if isinstance(a, dict)}

    for key in ("preferred_username", "email", "sub"):
        val = attrs.get(key)
        if isinstance(val, str) and val.strip():
            return val

    raise HTTPException(status_code=500, detail="Impossibile determinare lo username dall'user-info")

def get_username_from_access_token(sdk: CognitoSDK, access_token: str) -> str:
    """
    Helper “one-shot”: chiama Cognito e restituisce direttamente lo username.
    """
    userinfo = fetch_user_info_by_token(sdk, access_token)
    return extract_username_from_userinfo(userinfo)
