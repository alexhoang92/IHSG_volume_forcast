import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-production-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "none")  # "lax" for localhost dev
COOKIE_SECURE = COOKIE_SAMESITE == "none"  # secure must be True when samesite=none

router = APIRouter()


def _create_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(access_token: str | None = Cookie(default=None)) -> str:
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub", "")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@router.post("/login")
def login(response: Response, form: OAuth2PasswordRequestForm = Depends()):
    if form.username != "ajaib" or form.password != "admin":
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = _create_token(form.username)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,  # type: ignore[arg-type]
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
    return {"message": "Login successful"}


@router.post("/logout")
def logout(response: Response, _: str = Depends(get_current_user)):
    response.delete_cookie(
        key="access_token",
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,  # type: ignore[arg-type]
    )
    return {"message": "Logged out"}
