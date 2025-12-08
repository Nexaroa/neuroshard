from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import requests
import asyncio
from typing import Optional, List, Dict
from dotenv import load_dotenv
from . import models, schemas, database, auth_utils, dependencies, downloads, ledger
from . import waitlist as waitlist_module
from .wallet import wallet_manager
from pydantic import BaseModel
import random
import os
import uuid
import math
import hashlib

from neuroshard.core.economics import calculate_stake_multiplier

# ECDSA node_id derivation (matches neuroshard/core/crypto.py)
def derive_ecdsa_node_id(token: str) -> str:
    """
    Derive ECDSA node_id from token.
    
    This matches the derivation in neuroshard/core/crypto.py:
    1. private_key = SHA256(token)
    2. public_key = ECDSA_derive(private_key) on secp256k1
    3. node_id = SHA256(public_key)[:32]
    
    Since we don't want to import the full crypto module here,
    we compute it directly using the cryptography library.
    """
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    
    # Derive private key from token
    private_key_bytes = hashlib.sha256(token.encode()).digest()
    
    # Create ECDSA private key
    private_key = ec.derive_private_key(
        int.from_bytes(private_key_bytes, 'big'),
        ec.SECP256K1(),
        default_backend()
    )
    
    # Get compressed public key
    public_key = private_key.public_key()
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.CompressedPoint
    )
    
    # node_id = SHA256(public_key)[:32]
    return hashlib.sha256(public_key_bytes).hexdigest()[:32]

# Load environment variables
load_dotenv()

# Initialize SQLite Database for Users (Decoupled from Tracker)
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

# Configure CORS
cors_origins = os.getenv("CORS_ORIGINS", "https://neuroshard.com,http://localhost:5173,http://localhost:8000,http://127.0.0.1:5173").split(",")
allow_origins = [origin.strip() for origin in cors_origins if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(downloads.router, prefix="/api/downloads", tags=["downloads"])
app.include_router(ledger.router)
app.include_router(waitlist_module.router)

# DECENTRALIZED: No more central reward loop hitting Postgres or Tracker
# Credits are now handled by the LedgerManager on individual nodes.
# The website database (sqlite/postgres) only stores "purchased" credits or initial grants.
# Real-time mining rewards live in the node's local wallet.

@app.on_event("startup")
async def startup_event():
    # No reward loop needed
    pass

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

@app.post("/api/chat")
async def chat_proxy(
    req: ChatRequest,
    current_user: models.User = Depends(dependencies.get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Proxy chat requests to an available Entry Node in the swarm. 
    Requires Authentication and NEURO balance.
    
    Fee Structure (from economics.py):
    - Cost: 0.1 NEURO per 1M tokens (INFERENCE_REWARD_PER_MILLION)
    - Fee: 5% burned (deflationary)
    """
    try:
        # Check NEURO balance from ledger
        if not current_user.node_id:
            raise HTTPException(
                status_code=400,
                detail="Wallet required. Please create or connect a wallet in the dashboard."
            )
        
        # Get NEURO balance from new NEUROLedger format
        import sqlite3
        node_id = current_user.node_id
        
        # Check LEDGER_DB_PATH first, then construct from LEDGER_DATA_DIR
        ledger_db_path = os.getenv("LEDGER_DB_PATH")
        if not ledger_db_path:
            ledger_data_dir = os.getenv("LEDGER_DATA_DIR", "/data")
            ledger_db_path = os.path.join(ledger_data_dir, "node_ledger.db")
        
        # Fee burn constants (from whitepaper)
        FEE_BURN_RATE = 0.05  # 5% burned
        BURN_ADDRESS = "BURN_0x0000000000000000000000000000000000000000"
        
        neuro_balance = 0.0
        if os.path.exists(ledger_db_path):
            try:
                conn = sqlite3.connect(ledger_db_path, check_same_thread=False)
                cursor = conn.cursor()
                
                # Try new 'balances' table first (NEUROLedger format)
                try:
                    cursor.execute("SELECT balance FROM balances WHERE node_id = ?", (node_id,))
                    row = cursor.fetchone()
                    if row:
                        neuro_balance = row[0]
                except sqlite3.OperationalError:
                    # Fall back to legacy 'credits' table
                    cursor.execute("SELECT balance FROM credits WHERE node_id = ?", (node_id,))
                    row = cursor.fetchone()
                    if row:
                        neuro_balance = row[0]
                
                conn.close()
            except Exception as e:
                print(f"Ledger read error: {e}")
        
        # Pricing Model: Based on economics.py - INFERENCE_REWARD_PER_MILLION per 1M tokens + 5% fee
        # Import from economics to keep pricing consistent
        try:
            from neuroshard.core.economics import INFERENCE_REWARD_PER_MILLION
            price_per_million = INFERENCE_REWARD_PER_MILLION  # 0.1 NEURO per 1M tokens
        except ImportError:
            price_per_million = 0.1  # Fallback
        
        prompt_tokens = len(req.prompt) // 4
        total_estimated_tokens = prompt_tokens + req.max_new_tokens
        base_cost = (total_estimated_tokens / 1_000_000.0) * price_per_million
        base_cost = max(0.00001, base_cost)  # Minimum cost (lowered)
        
        fee = base_cost * FEE_BURN_RATE
        total_cost = base_cost + fee
        
        if neuro_balance < total_cost:
             raise HTTPException(
                 status_code=402, 
                 detail=f"Insufficient NEURO. Request costs {total_cost:.6f} NEURO ({total_estimated_tokens} tokens + 5% fee), but you have {neuro_balance:.6f} NEURO. Keep your node running to earn more!"
             )

        # Find Entry Nodes (Decentralized Discovery)
        # Entry nodes are nodes that have layer 0 (embedding layer)
        tracker_url = os.getenv("TRACKER_URL", "http://tracker:3000")
        
        try:
            peers_resp = requests.get(f"{tracker_url}/peers", params={"layer_needed": 0}, timeout=2)
            if peers_resp.status_code == 200:
                peers = peers_resp.json()
                entry_nodes = [p['url'] for p in peers]
            else:
                entry_nodes = []
        except Exception as e:
            print(f"Tracker error: {e}")
            entry_nodes = []
            
        if not entry_nodes:
             raise HTTPException(status_code=503, detail="No Entry Nodes available in the swarm. Please wait for nodes to register.")
            
        # Pick random entry node
        target_node = random.choice(entry_nodes)
        
        # Try multiple nodes with retry logic for robustness
        last_error = None
        random.shuffle(entry_nodes)  # Randomize order for load balancing
        
        for target_node in entry_nodes[:3]:  # Try up to 3 nodes
            # HACK: For local docker testing
            if "localhost" in target_node or "127.0.0.1" in target_node:
                 target_node = target_node.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")

            print(f"Forwarding to Node: {target_node}")

            try:
                # Forward request with longer timeout for inference
                resp = requests.post(f"{target_node}/generate_text", json={
                    "prompt": req.prompt,
                    "max_new_tokens": req.max_new_tokens
                }, timeout=30)  # 30 seconds for inference
                
                if resp.status_code == 200:
                    break  # Success, exit retry loop
                else:
                    last_error = f"Node returned {resp.status_code}"
                    print(f"Node {target_node} returned {resp.status_code}, trying next...")
                    continue
            except requests.exceptions.Timeout:
                last_error = f"Node {target_node} timed out"
                print(f"Node {target_node} timed out, trying next...")
                continue
            except requests.exceptions.ConnectionError as e:
                last_error = f"Could not connect to {target_node}"
                print(f"Could not connect to {target_node}: {e}, trying next...")
                continue
            except Exception as e:
                last_error = str(e)
                print(f"Error with {target_node}: {e}, trying next...")
                continue
        else:
            # All nodes failed
            raise HTTPException(
                status_code=503, 
                detail=f"All entry nodes failed. Last error: {last_error}"
            ) 
        
        if resp.status_code == 200:
            # Deduct NEURO with fee burn (deflationary mechanism)
            if os.path.exists(ledger_db_path):
                 try:
                     conn = sqlite3.connect(ledger_db_path, check_same_thread=False)
                     cursor = conn.cursor()
                     
                     # Check which table format exists
                     cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='balances'")
                     use_new_format = cursor.fetchone() is not None
                     
                     if use_new_format:
                         # New NEUROLedger format with fee burn
                         # 1. Deduct total cost from user
                         cursor.execute("""
                             UPDATE balances SET 
                                 balance = balance - ?,
                                 total_spent = COALESCE(total_spent, 0) + ?
                             WHERE node_id = ?
                         """, (total_cost, total_cost, node_id))
                         
                         # 2. Credit fee to burn address (deflationary)
                         cursor.execute("""
                             INSERT INTO balances (node_id, balance, total_earned, created_at)
                             VALUES (?, ?, ?, ?)
                             ON CONFLICT(node_id) DO UPDATE SET
                                 balance = balance + ?
                         """, (BURN_ADDRESS, fee, fee, __import__('time').time(), fee))
                         
                         # 3. Update global burn stats
                         cursor.execute("""
                             UPDATE global_stats SET
                                 total_burned = COALESCE(total_burned, 0) + ?,
                                 updated_at = ?
                             WHERE id = 1
                         """, (fee, __import__('time').time()))
                         
                         print(f"NEURO spent: {base_cost:.6f} + {fee:.6f} fee (burned)")
                     else:
                         # Legacy format - just deduct
                         cursor.execute("UPDATE credits SET balance = balance - ? WHERE node_id = ?", (total_cost, node_id))
                     
                     conn.commit()
                     conn.close()
                 except Exception as e:
                     print(f"Ledger deduct error: {e}")

            return resp.json()
        else:
            raise HTTPException(status_code=502, detail=f"Node Error: {resp.text}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat Proxy Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/token", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth_utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=auth_utils.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_utils.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    # Create refresh token with unique ID for revocation support
    token_id = auth_utils.generate_refresh_token_id()
    refresh_token_expires = timedelta(days=auth_utils.REFRESH_TOKEN_EXPIRE_DAYS)
    refresh_token = auth_utils.create_refresh_token(
        data={"sub": user.email, "token_id": token_id}, expires_delta=refresh_token_expires
    )
    
    # Store refresh token in database for revocation tracking
    db_refresh_token = models.RefreshToken(
        token_id=token_id,
        user_id=user.id,
        expires_at=datetime.utcnow() + refresh_token_expires
    )
    db.add(db_refresh_token)
    
    # Update last login timestamp
    user.last_login = datetime.utcnow()
    db.commit()
    
    return {
        "access_token": access_token, 
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": auth_utils.ACCESS_TOKEN_EXPIRE_MINUTES * 60  # seconds
    }


@app.post("/api/auth/token/refresh", response_model=schemas.Token)
async def refresh_access_token(
    request: schemas.RefreshTokenRequest,
    db: Session = Depends(database.get_db)
):
    """
    Refresh an access token using a valid refresh token.
    
    This allows clients to get a new access token without re-authenticating
    with username/password, as long as the refresh token is still valid.
    """
    # Verify the refresh token JWT
    payload = auth_utils.verify_refresh_token(request.refresh_token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    email = payload.get("sub")
    token_id = payload.get("token_id")
    
    if not email or not token_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if token exists and is not revoked
    db_token = db.query(models.RefreshToken).filter(
        models.RefreshToken.token_id == token_id
    ).first()
    
    if not db_token or db_token.revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get the user
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create new access token
    access_token_expires = timedelta(minutes=auth_utils.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_utils.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    # Rotate refresh token (create new one, revoke old one)
    new_token_id = auth_utils.generate_refresh_token_id()
    refresh_token_expires = timedelta(days=auth_utils.REFRESH_TOKEN_EXPIRE_DAYS)
    new_refresh_token = auth_utils.create_refresh_token(
        data={"sub": user.email, "token_id": new_token_id}, expires_delta=refresh_token_expires
    )
    
    # Revoke old token and create new one
    db_token.revoked = True
    new_db_token = models.RefreshToken(
        token_id=new_token_id,
        user_id=user.id,
        expires_at=datetime.utcnow() + refresh_token_expires
    )
    db.add(new_db_token)
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
        "expires_in": auth_utils.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }


@app.post("/api/auth/logout")
async def logout(
    request: schemas.RefreshTokenRequest,
    db: Session = Depends(database.get_db)
):
    """
    Logout by revoking the refresh token.
    
    This prevents the refresh token from being used to generate new access tokens.
    The client should also clear any stored tokens locally.
    """
    payload = auth_utils.verify_refresh_token(request.refresh_token)
    if payload:
        token_id = payload.get("token_id")
        if token_id:
            db_token = db.query(models.RefreshToken).filter(
                models.RefreshToken.token_id == token_id
            ).first()
            if db_token:
                db_token.revoked = True
                db.commit()
    
    return {"message": "Successfully logged out"}

@app.post("/api/auth/signup", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    """
    Create new user account.
    
    WAITLIST FLOW:
    - User must first join waitlist and be approved
    - Upon approval, they receive an email with signup link
    - This endpoint checks waitlist status before allowing account creation
    
    NOTE: This does NOT create a wallet automatically.
    User must call /wallet/create or /wallet/connect after signup.
    """
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Check if user is on the waitlist and approved
    waitlist_entry = db.query(models.WaitlistEntry).filter(
        models.WaitlistEntry.email == user.email
    ).first()
    
    if not waitlist_entry:
        raise HTTPException(
            status_code=403,
            detail="You must join the waitlist first. Please register your hardware at /join"
        )
    
    if waitlist_entry.status == "pending":
        raise HTTPException(
            status_code=403,
            detail="Your waitlist application is still pending approval. We'll email you when you're approved."
        )
    
    if waitlist_entry.status == "rejected":
        raise HTTPException(
            status_code=403,
            detail="Your waitlist application was not approved. Please contact support."
        )
    
    if waitlist_entry.status == "converted":
        raise HTTPException(
            status_code=400,
            detail="This waitlist entry has already been used to create an account."
        )
    
    # User is approved - allow account creation
    hashed_password = auth_utils.get_password_hash(user.password)
    
    db_user = models.User(
        email=user.email, 
        hashed_password=hashed_password,
        node_id=None,  # Will be set when wallet is created/connected
        wallet_id=None,
        waitlist_approved=True,
        waitlist_id=waitlist_entry.id
    )
    db.add(db_user)
    
    # Mark waitlist entry as converted
    waitlist_entry.status = "converted"
    waitlist_entry.converted_at = datetime.utcnow()
    
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/api/wallet/create", response_model=schemas.WalletCreate)
async def create_wallet(
    current_user: models.User = Depends(dependencies.get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Generate a NEW wallet with BIP39 mnemonic seed phrase.
    
    ⚠️  CRITICAL: The mnemonic is shown ONLY ONCE!
    User MUST save it - we don't store private keys in the database.
    
    Similar to MetaMask wallet creation.
    """
    # Check if user already has a wallet
    if current_user.node_id:
        raise HTTPException(
            status_code=400,
            detail="Wallet already exists. Use /wallet/recover to import a different wallet."
        )
    
    # Generate new wallet
    wallet = wallet_manager.create_wallet()
    
    # Save ONLY public info to database
    current_user.node_id = wallet['node_id']
    current_user.wallet_id = wallet['wallet_id']
    db.commit()
    
    # Return everything INCLUDING the mnemonic (shown only this once!)
    return schemas.WalletCreate(**wallet)

@app.post("/api/wallet/connect", response_model=schemas.WalletInfo)
async def connect_wallet(
    wallet_data: schemas.WalletConnect,
    current_user: models.User = Depends(dependencies.get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Connect/Import wallet using mnemonic seed phrase or node token.
    
    This allows users to:
    - Import existing wallet from another device
    - Recover wallet from backup
    - Switch wallets
    
    Similar to MetaMask "Import Wallet" feature.
    """
    try:
        # Try to recover wallet from the secret (mnemonic or token)
        secret = wallet_data.secret.strip()
        
        # Check if it's a mnemonic (12 words) or a token (hex string)
        if len(secret.split()) == 12:
            # It's a mnemonic
            wallet = wallet_manager.recover_wallet(secret)
        else:
            # It's a raw token - derive node_id
            wallet = {
                'token': secret,
                'node_id': wallet_manager.token_to_node_id(secret),
                'wallet_id': wallet_manager.token_to_node_id(secret)[:16]
            }
        
        # Check if this wallet is already used by another user
        existing_user = db.query(models.User).filter(
            models.User.node_id == wallet['node_id'],
            models.User.id != current_user.id
        ).first()
        
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="This wallet is already connected to another account"
            )
        
        # Save ONLY public info to database
        current_user.node_id = wallet['node_id']
        current_user.wallet_id = wallet['wallet_id']
        db.commit()
        
        # Return only public info
        return schemas.WalletInfo(
            node_id=wallet['node_id'],
            wallet_id=wallet['wallet_id'],
            balance=0.0  # Will be fetched from ledger
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to connect wallet: {str(e)}")

@app.get("/api/users/me/wallet")
async def read_user_wallet(current_user: models.User = Depends(dependencies.get_current_user)):
    """Get user's public wallet info (no private keys)"""
    if not current_user.node_id:
        return {
            "connected": False,
            "node_id": None,
            "wallet_id": None,
            "message": "No wallet connected. Create or import a wallet."
        }
    
    return {
        "connected": True,
        "node_id": current_user.node_id,
        "wallet_id": current_user.wallet_id
    }

# Legacy endpoint for backwards compatibility
@app.get("/api/users/me/token")
async def read_user_token_legacy(current_user: models.User = Depends(dependencies.get_current_user)):
    """
    DEPRECATED: Use /users/me/wallet instead.
    This endpoint no longer returns the private token (security improvement).
    """
    return {
        "node_id": current_user.node_id,
        "wallet_id": current_user.wallet_id,
        "deprecated": True,
        "message": "Private keys are no longer stored in database. Use /users/me/wallet"
    }

@app.get("/api/users/me", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(dependencies.get_current_user)):
    return current_user

@app.get("/api/stats")
async def get_global_stats():
    """Proxy request to the central tracker to get network stats."""
    try:
        tracker_url = os.getenv("TRACKER_URL", "http://tracker:3000")
        response = requests.get(f"{tracker_url}/stats", timeout=2)
        if response.status_code == 200:
            return response.json()
        else:
            return {"active_nodes": 0, "model_size": "142B", "total_tps": 0, "avg_latency": "N/A"}
    except Exception as e:
        return {"active_nodes": 0, "model_size": "142B", "total_tps": 0, "avg_latency": "N/A"}

@app.get("/api/admin/peers")
async def get_admin_peers(
    current_admin: models.User = Depends(dependencies.get_current_admin_user)
):
    """
    DECENTRALIZED CRAWLER - Admin Only
    In a fully decentralized network, there is no central list of all peers.
    To visualize the network for the admin, we must 'crawl' it starting from bootstrap nodes.
    """

    # 1. Start with bootstrap list from tracker
    known_peers = set()
    try:
        tracker_url = os.getenv("TRACKER_URL", "http://tracker:3000")
        response = requests.get(f"{tracker_url}/peers", timeout=2)
        if response.status_code == 200:
            initial_peers = response.json()
            for p in initial_peers:
                known_peers.add(p['url'])
    except:
        pass
    
    # 2. (Optional) Crawl one level deep if list is small
    # For now, we just return the bootstrap list + any cached nodes from previous crawls
    # Real crawler would run in background and cache to Redis/File
    
    # Format for frontend
    results = []
    for url in known_peers:
        # We mock stats if we can't reach them, or ping them live
        results.append({
            "url": url,
            "shard_range": "unknown", # Would need to ping to get actuals
            "last_seen": 0,
            "tps": 0,
            "latency": 0,
            "is_entry": False,
            "is_exit": False
        })
        
    return results


# =============================================================================
# ADMIN USER MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/api/admin/users")
async def get_admin_users(
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """
    Get all registered users - Admin Only.
    Returns user list with registration info.
    """
    users = db.query(models.User).order_by(models.User.created_at.desc()).all()
    
    return [
        {
            "id": user.id,
            "email": user.email,
            "is_active": user.is_active,
            "is_admin": user.is_admin,
            "has_wallet": user.node_id is not None,
            "wallet_id": user.wallet_id if user.node_id else None,
            "node_id": user.node_id if user.node_id else None,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
        for user in users
    ]


@app.get("/api/admin/users/{user_id}")
async def get_admin_user_detail(
    user_id: int,
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """Get detailed info about a specific user - Admin Only."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "email": user.email,
        "is_active": user.is_active,
        "is_admin": user.is_admin,
        "node_id": user.node_id,
        "wallet_id": user.wallet_id,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None
    }


@app.patch("/api/admin/users/{user_id}/toggle-admin")
async def toggle_user_admin(
    user_id: int,
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """Toggle admin status for a user - Admin Only."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent removing own admin status
    if user.id == current_admin.id:
        raise HTTPException(status_code=400, detail="Cannot modify your own admin status")
    
    user.is_admin = not user.is_admin
    db.commit()
    
    return {
        "id": user.id,
        "email": user.email,
        "is_admin": user.is_admin,
        "message": f"User {'promoted to' if user.is_admin else 'demoted from'} admin"
    }


@app.patch("/api/admin/users/{user_id}/toggle-active")
async def toggle_user_active(
    user_id: int,
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """Toggle active status for a user - Admin Only."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent deactivating own account
    if user.id == current_admin.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
    
    user.is_active = not user.is_active
    db.commit()
    
    return {
        "id": user.id,
        "email": user.email,
        "is_active": user.is_active,
        "message": f"User {'activated' if user.is_active else 'deactivated'}"
    }


@app.delete("/api/admin/users/{user_id}")
async def delete_user(
    user_id: int,
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """
    Permanently delete a user - Admin Only.
    This also deletes associated refresh tokens.
    """
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent deleting own account
    if user.id == current_admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    # Store email for response
    deleted_email = user.email
    
    # Delete associated refresh tokens first
    db.query(models.RefreshToken).filter(models.RefreshToken.user_id == user_id).delete()
    
    # Delete the user
    db.delete(user)
    db.commit()
    
    return {
        "success": True,
        "message": f"User {deleted_email} has been permanently deleted"
    }


@app.get("/api/admin/stats")
async def get_admin_stats(
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """Get admin dashboard statistics - Admin Only."""
    total_users = db.query(models.User).count()
    active_users = db.query(models.User).filter(models.User.is_active == True).count()
    admin_users = db.query(models.User).filter(models.User.is_admin == True).count()
    users_with_wallets = db.query(models.User).filter(models.User.node_id != None).count()
    
    return {
        "total_users": total_users,
        "users_with_wallets": users_with_wallets,
        "active_users": active_users,
        "admin_users": admin_users
    }


@app.get("/api/users/me/is_admin")
async def check_is_admin(current_user: models.User = Depends(dependencies.get_current_user)):
    """Check if current user is an admin."""
    return {"is_admin": current_user.is_admin}


@app.get("/api/node/neuro")
async def get_node_neuro(node_id: str = None, token: str = None):
    """
    Get NEURO token balance and stats from the distributed ledger.
    
    Can accept either:
    - node_id: Public wallet address (preferred)
    - token: DEPRECATED - Private token (for backwards compatibility)
    
    Returns full account info including:
    - balance: Current spendable balance
    - total_earned: Lifetime earnings
    - total_spent: Lifetime spending
    - stake: Currently staked amount
    - stake_multiplier: Reward multiplier from staking
    """
    import sqlite3
    import time as time_module
    
    # Accept either node_id or token (derive node_id from token if needed)
    if not node_id and not token:
        raise HTTPException(status_code=400, detail="Either node_id or token required")
    
    if token and not node_id:
        # Legacy: Derive ECDSA node_id from token
        node_id = derive_ecdsa_node_id(token)
    
    # Try to query ledger database (if available on server)
    # Check LEDGER_DB_PATH first, then construct from LEDGER_DATA_DIR
    ledger_db_path = os.getenv("LEDGER_DB_PATH")
    if not ledger_db_path:
        ledger_data_dir = os.getenv("LEDGER_DATA_DIR", "/data")
        ledger_db_path = os.path.join(ledger_data_dir, "node_ledger.db")
    
    # Account info
    balance = 0.0
    total_earned = 0.0
    total_spent = 0.0
    stake = 0.0
    stake_multiplier = 1.0
    stake_locked_until = 0.0
    proof_count = 0
    source = "ledger"
    
    # Global stats
    total_burned = 0.0
    circulating_supply = 0.0

    if os.path.exists(ledger_db_path):
        try:
            conn = sqlite3.connect(ledger_db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # Check which table format exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='balances'")
            use_new_format = cursor.fetchone() is not None
            
            if use_new_format:
                # New NEUROLedger format
                cursor.execute("""
                    SELECT balance, total_earned, total_spent, proof_count 
                    FROM balances WHERE node_id = ?
                """, (node_id,))
                row = cursor.fetchone()
                if row:
                    balance, total_earned, total_spent, proof_count = row
                    total_earned = total_earned or 0.0
                    total_spent = total_spent or 0.0
                    proof_count = proof_count or 0
                
                # Get stake info
                cursor.execute("""
                    SELECT amount, locked_until FROM stakes WHERE node_id = ?
                """, (node_id,))
                stake_row = cursor.fetchone()
                if stake_row:
                    stake_amount = stake_row[0]
                    stake_locked_until = stake_row[1] or 0.0
                    # Only set stake if it's actually a positive value
                    if stake_amount is not None and stake_amount > 0:
                        stake = float(stake_amount)
                        # Calculate multiplier with DIMINISHING RETURNS (from economics module)
                        # Multiplier applies even after unlock (until unstaked)
                        stake_multiplier = calculate_stake_multiplier(stake)
                    else:
                        # Ensure stake is 0 and multiplier is 1.0 if no valid stake
                        stake = 0.0
                        stake_multiplier = 1.0
                        stake_locked_until = 0.0
                
                # Get global burn stats
                cursor.execute("""
                    SELECT total_minted, total_burned FROM global_stats WHERE id = 1
                """)
                stats_row = cursor.fetchone()
                if stats_row:
                    total_minted = stats_row[0] or 0.0
                    total_burned = stats_row[1] or 0.0
                    circulating_supply = total_minted - total_burned
                
                source = "neuro_ledger"
            else:
                # Legacy format
                cursor.execute("SELECT balance FROM credits WHERE node_id = ?", (node_id,))
                row = cursor.fetchone()
                if row:
                    balance = row[0]
                source = "legacy_ledger"
                
            conn.close()
        except Exception as e:
             source = f"error_db: {e}"
    else:
        source = "unavailable"
        
    # NOTE: Legacy tracker stake lookup removed
    # Stake info now comes from ledger (NEUROLedger.stakes table)
    # Tracker no longer stores private node tokens for security reasons

    # Ensure stake_multiplier is 1.0 if stake is 0 or None
    if stake is None or stake <= 0:
        stake = 0.0
        stake_multiplier = 1.0
        stake_locked_until = 0.0
    
    return {
        "neuro_balance": round(balance, 6),
        "total_earned": round(total_earned, 6),
        "total_spent": round(total_spent, 6),
        "staked_balance": round(stake, 2),
        "stake_multiplier": round(stake_multiplier, 2),
        "stake_locked_until": stake_locked_until if stake_locked_until > 0 else None,
        "proof_count": proof_count,
        "node_id": node_id,
        "source": source,
        # Global network stats
        "network": {
            "total_burned": round(total_burned, 6),
            "circulating_supply": round(circulating_supply, 6),
            "burn_rate": "5%"
        }
    }

@app.get("/api/training/global")
async def get_global_training_status():
    """
    Get global LLM training status from GOSSIP DATA (ledger).
    
    DECENTRALIZED: We don't query nodes directly (they may be behind NAT).
    Instead, we read from the observer's ledger which receives PoNW proofs via gossip.
    
    This shows whether the distributed training is actually working:
    - Is the model improving?
    - Are nodes converging to the same weights?
    - What's the network-wide loss?
    """
    import sqlite3
    import time
    
    # Default response structure
    training_status = {
        "is_training": False,
        "training_verified": False,
        "is_converging": True,
        "global_loss": 0.0,
        "loss_trend": "unknown",
        "hash_agreement_rate": 1.0,
        "total_nodes_training": 0,
        "total_training_steps": 0,
        "total_tokens_trained": 0,
        "data_shards_covered": 0,
        "sync_success_rate": 0.0,
        "diloco": {
            "enabled": True,
            "inner_steps_config": 500,
            "outer_steps_completed": 0,
        },
        "nodes": [],
        "source": "gossip"
    }
    
    try:
        # Read training data from observer's ledger (populated via gossip)
        # Check LEDGER_DB_PATH first, then LEDGER_DATA_DIR
        ledger_db_path = os.getenv("LEDGER_DB_PATH")
        if not ledger_db_path:
            ledger_data_dir = os.getenv("LEDGER_DATA_DIR", "/data")
            ledger_db_path = os.path.join(ledger_data_dir, "node_ledger.db")
        
        if not os.path.exists(ledger_db_path):
            training_status["source"] = f"ledger_not_found:{ledger_db_path}"
            return training_status
        
        conn = sqlite3.connect(ledger_db_path)
        cursor = conn.cursor()
        
        # Check if proof_history table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='proof_history'")
        if not cursor.fetchone():
            conn.close()
            training_status["source"] = "no_proof_history_table"
            return training_status
        
        # Get training proofs from last 5 minutes (recent activity)
        recent_cutoff = time.time() - 300  # 5 minutes
        
        # Aggregate training stats by node (table is proof_history, not ponw_proofs)
        cursor.execute("""
            SELECT 
                node_id, 
                SUM(training_batches) as total_batches,
                SUM(tokens_processed) as total_tokens,
                COUNT(*) as proof_count,
                MAX(timestamp) as last_seen
            FROM proof_history 
            WHERE timestamp > ? AND proof_type = 'training'
            GROUP BY node_id
            ORDER BY total_batches DESC
        """, (recent_cutoff,))
        
        training_nodes = []
        total_batches_all = 0
        total_tokens_all = 0
        loss_values = []
        
        for row in cursor.fetchall():
            node_id, total_batches, total_tokens, proof_count, last_seen = row
            batches = int(total_batches or 0)
            tokens = int(total_tokens or 0)
            
            # Get the most recent loss for this node (separate query)
            # Also check if this node is a validator (has_lm_head=1) for proper cross-entropy loss
            cursor.execute("""
                SELECT current_loss, has_lm_head FROM proof_history 
                WHERE node_id = ? AND proof_type = 'training' AND current_loss IS NOT NULL
                ORDER BY timestamp DESC LIMIT 1
            """, (node_id,))
            loss_row = cursor.fetchone()
            latest_loss = loss_row[0] if loss_row else None
            is_validator = bool(loss_row[1]) if loss_row and len(loss_row) > 1 else False
            
            training_nodes.append({
                "node_id": (node_id or "unknown")[:12],
                "training_rounds": batches,
                "proofs_submitted": proof_count,
                "last_active": last_seen,
                "current_loss": latest_loss,
                "is_validator": is_validator,
            })
            total_batches_all += batches
            total_tokens_all += tokens
            # Include all valid loss values (validators have true cross-entropy, workers have proxy loss)
            if latest_loss is not None and latest_loss > 0:
                loss_values.append(latest_loss)
        
        # Get total training steps from all time
        cursor.execute("""
            SELECT COUNT(DISTINCT node_id), SUM(training_batches), SUM(tokens_processed)
            FROM proof_history 
            WHERE proof_type = 'training'
        """)
        row = cursor.fetchone()
        all_time_nodes = row[0] or 0
        all_time_batches = int(row[1] or 0)
        all_time_tokens = int(row[2] or 0)
        
        # Get unique proofs from recent activity
        cursor.execute("""
            SELECT COUNT(DISTINCT signature) 
            FROM proof_history 
            WHERE timestamp > ?
        """, (recent_cutoff,))
        unique_proofs = cursor.fetchone()[0] or 0
        
        # Get total proofs count from global_stats (actual schema)
        max_epoch = 0
        try:
            cursor.execute("SELECT total_proofs FROM global_stats WHERE id = 1")
            row = cursor.fetchone()
            if row:
                # Estimate outer steps: total_proofs / expected_proofs_per_sync
                total_proofs = int(row[0] or 0)
                # Each outer step = ~500 inner steps = ~8 proofs (assuming 60s proof interval)
                max_epoch = total_proofs // 8 if total_proofs > 0 else 0
        except Exception as e:
            pass  # Table might not exist or have different schema
        
        # Count unique data shards (based on unique node_ids that have done training)
        try:
            cursor.execute("SELECT COUNT(DISTINCT node_id) FROM proof_history WHERE proof_type = 'training'")
            data_shards = cursor.fetchone()[0] or 0
        except:
            data_shards = 0
        
        conn.close()
        
        # Populate response
        training_status["total_nodes_training"] = len(training_nodes)
        training_status["is_training"] = len(training_nodes) > 0
        training_status["total_training_steps"] = all_time_batches
        training_status["total_tokens_trained"] = all_time_tokens
        training_status["data_shards_covered"] = data_shards
        training_status["nodes"] = training_nodes[:10]
        
        # Calculate global loss (average of all nodes with valid loss)
        if loss_values:
            training_status["global_loss"] = round(sum(loss_values) / len(loss_values), 4)
            # Determine trend based on loss value (typical cross-entropy range)
            if training_status["global_loss"] < 2.0:
                training_status["loss_trend"] = "improving"
            elif training_status["global_loss"] < 5.0:
                training_status["loss_trend"] = "stable"
            else:
                training_status["loss_trend"] = "learning"
        
        # Training is verified if we have recent training proofs
        if len(training_nodes) > 0:
            training_status["training_verified"] = True
            training_status["is_converging"] = len(loss_values) > 0  # Converging if we have loss data
            if not loss_values:
                # Training is happening but no loss data yet
                training_status["loss_trend"] = "initializing"
        
        # DiLoCo info
        training_status["diloco"]["outer_steps_completed"] = max_epoch
        steps_until_sync = 500 - (all_time_batches % 500) if all_time_batches > 0 else 500
        training_status["diloco"]["steps_until_sync"] = steps_until_sync
        
        # Calculate sync success rate based on recent proof activity
        if len(training_nodes) > 0:
            # Success rate = unique proofs in last 5 min / expected proofs (1 per node per minute)
            expected_proofs = len(training_nodes) * 5  # 5 minutes * 1 proof/minute/node
            training_status["sync_success_rate"] = min(1.0, unique_proofs / max(1, expected_proofs))
        
        training_status["source"] = "gossip_ledger"
        
    except Exception as e:
        training_status["source"] = f"error: {str(e)}"
    
    return training_status


@app.get("/api/neuro/stats")
async def get_neuro_global_stats():
    """
    Get global NEURO token statistics.
    
    Returns:
    - total_minted: Total NEURO ever created through PoNW
    - total_burned: Total NEURO burned through fee mechanism
    - circulating_supply: total_minted - total_burned
    - burn_rate: Current burn rate (5%)
    - total_proofs: Number of PoNW proofs processed
    - total_transactions: Number of transfers
    """
    import sqlite3
    
    # Check LEDGER_DB_PATH first, then construct from LEDGER_DATA_DIR
    ledger_db_path = os.getenv("LEDGER_DB_PATH")
    if not ledger_db_path:
        ledger_data_dir = os.getenv("LEDGER_DATA_DIR", "/data")
        ledger_db_path = os.path.join(ledger_data_dir, "node_ledger.db")
    
    stats = {
        "total_minted": 0.0,
        "total_burned": 0.0,
        "circulating_supply": 0.0,
        "total_proofs": 0,
        "total_transactions": 0,
        "burn_rate": "5%",
        "source": "unavailable"
    }
    
    if os.path.exists(ledger_db_path):
        try:
            conn = sqlite3.connect(ledger_db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # Check for new format
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='global_stats'")
            if cursor.fetchone():
                cursor.execute("""
                    SELECT total_minted, total_burned, total_transferred, total_proofs, total_transactions
                    FROM global_stats WHERE id = 1
                """)
                row = cursor.fetchone()
                if row:
                    total_minted = row[0] or 0.0
                    total_burned = row[1] or 0.0
                    stats["total_minted"] = round(total_minted, 6)
                    stats["total_burned"] = round(total_burned, 6)
                    stats["circulating_supply"] = round(total_minted - total_burned, 6)
                    stats["total_proofs"] = row[3] or 0
                    stats["total_transactions"] = row[4] or 0
                    stats["source"] = "neuro_ledger"
            else:
                # Legacy - estimate from credits table
                cursor.execute("SELECT SUM(balance) FROM credits")
                row = cursor.fetchone()
                if row and row[0]:
                    stats["circulating_supply"] = round(row[0], 6)
                    stats["total_minted"] = stats["circulating_supply"]  # No burn in legacy
                stats["source"] = "legacy_ledger"
            
            conn.close()
        except Exception as e:
            stats["source"] = f"error: {e}"
    
    return stats


@app.get("/api/users/me/node_status")
async def get_my_node_status(current_user: models.User = Depends(dependencies.get_current_user)):
    """Check if the current user's node is active."""
    if not current_user.node_id:
        return {"active": False, "detail": "No wallet connected"}
        
    # Check if user's node_id appears in active peers
    tracker_url = os.getenv("TRACKER_URL", "http://tracker:3000")
    try:
        resp = requests.get(f"{tracker_url}/peers", timeout=2)
        if resp.status_code == 200:
            peers = resp.json()
            
            # Check each peer's node_token to see if it matches this user's node_id
            # node_id = derive_from(node_token), so we check if any peer's derived node_id matches
            from .wallet import WalletManager
            wallet_manager = WalletManager()
            
            for peer in peers:
                peer_token = peer.get("node_token")
                if peer_token:
                    try:
                        # Derive node_id from peer's token
                        peer_node_id = wallet_manager.token_to_node_id(peer_token)
                        if peer_node_id == current_user.node_id:
                            return {"active": True, "node_id": current_user.node_id}
                    except:
                        pass  # Skip peers with invalid tokens
            
            return {"active": False, "node_id": current_user.node_id, "detail": "Node not running"}
        else:
            return {"active": False, "detail": "Tracker error"}
    except Exception as e:
        logger.error(f"Node status check failed: {e}")
        return {"active": False, "detail": "Tracker unreachable"}


# =============================================================================
# PROTECTED WHITEPAPER ENDPOINT
# =============================================================================

from fastapi.responses import FileResponse

@app.get("/api/whitepaper/pdf")
async def get_whitepaper_pdf(current_user: models.User = Depends(dependencies.get_current_user)):
    """
    Serve the whitepaper PDF to authenticated users only.
    
    This protects our technical documentation from public access while
    allowing registered users to view and download it.
    """
    # Path to the whitepaper PDF (stored in protected location)
    whitepaper_path = os.path.join(os.path.dirname(__file__), "..", "protected", "whitepaper.pdf")
    
    # Fallback to public location if protected doesn't exist yet
    if not os.path.exists(whitepaper_path):
        whitepaper_path = os.path.join(os.path.dirname(__file__), "..", "public", "whitepaper.pdf")
    
    if not os.path.exists(whitepaper_path):
        raise HTTPException(status_code=404, detail="Whitepaper not found")
    
    return FileResponse(
        whitepaper_path,
        media_type="application/pdf",
        filename="NeuroShard_Whitepaper.pdf",
        headers={
            "Content-Disposition": "inline; filename=NeuroShard_Whitepaper.pdf"
        }
    )

@app.get("/api/whitepaper/info")
async def get_whitepaper_info(current_user: models.User = Depends(dependencies.get_current_user)):
    """Get metadata about the whitepaper."""
    return {
        "title": "NeuroShard: A Decentralized Architecture for Collective Intelligence",
        "version": "1.0",
        "date": "November 2025",
        "authors": "LZ",
        "sections": [
            "Introduction",
            "NeuroLLM Architecture",
            "Decentralized Training",
            "System Architecture",
            "Proof of Neural Work (PoNW)",
            "Robustness and Anti-Poisoning",
            "Governance and The NeuroDAO",
            "NEURO Token Economics",
            "Security Considerations",
            "Checkpoint System",
            "Implementation",
            "Vision and Roadmap"
        ],
        "access_level": "registered_users_only"
    }
