from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey, Float, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)  # Admin flag
    
    # Wallet - PUBLIC ONLY (never store private keys!)
    # node_id is the public wallet address (32-char hex from ECDSA public key)
    # This is derived from the user's mnemonic/token but is safe to store
    node_id = Column(String, unique=True, index=True, nullable=True)
    wallet_id = Column(String, nullable=True)  # First 16 chars of node_id (display only)
    
    # Waitlist status - users must be approved before they can create wallets
    waitlist_approved = Column(Boolean, default=False)
    waitlist_id = Column(Integer, ForeignKey("waitlist_entries.id"), nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    last_login = Column(DateTime, nullable=True)
    
    # Relationship to refresh tokens
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    
    # Relationship to waitlist entry
    waitlist_entry = relationship("WaitlistEntry", back_populates="user", foreign_keys=[waitlist_id])


class RefreshToken(Base):
    """
    Store refresh tokens for secure token management.
    Allows for token revocation and tracking active sessions.
    """
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token_id = Column(String, unique=True, index=True)  # Unique identifier for the token
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    revoked = Column(Boolean, default=False)
    
    # Relationship back to user
    user = relationship("User", back_populates="refresh_tokens")


class WaitlistEntry(Base):
    """
    Waitlist entries for users who want to join the NeuroShard network.
    Collects hardware specs and generates referral codes.
    Users can submit multiple applications (e.g., different hardware).
    """
    __tablename__ = "waitlist_entries"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True)  # Not unique - allow multiple entries per email
    
    # Hardware specifications
    gpu_model = Column(String, nullable=True)  # e.g., "RTX 4090", "M2 Pro", "None"
    gpu_vram = Column(Integer, nullable=True)  # VRAM in GB
    ram_gb = Column(Integer, nullable=False)  # System RAM in GB
    internet_speed = Column(Integer, nullable=True)  # Mbps
    operating_system = Column(String, nullable=True)  # Windows, macOS, Linux
    
    # Calculated estimates
    estimated_daily_neuro = Column(Float, default=0.0)
    hardware_tier = Column(String, default="basic")  # basic, standard, pro, elite
    hardware_score = Column(Integer, default=0)  # 0-100 score
    
    # Referral system - "Neuro Link"
    referral_code = Column(String, unique=True, index=True)  # Unique 8-char code
    referred_by = Column(String, nullable=True)  # referral_code of referrer
    referral_count = Column(Integer, default=0)  # Number of successful referrals
    referral_bonus_percent = Column(Float, default=0.0)  # Bonus % from referrals
    
    # Status
    status = Column(String, default="pending")  # pending, approved, rejected, converted
    position = Column(Integer, nullable=True)  # Position in waitlist queue
    priority_score = Column(Integer, default=0)  # Higher = earlier access
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    approved_at = Column(DateTime, nullable=True)
    converted_at = Column(DateTime, nullable=True)  # When they completed full signup
    
    # Admin notes
    admin_notes = Column(Text, nullable=True)
    
    # Email tracking
    confirmation_email_sent = Column(Boolean, default=False)
    approval_email_sent = Column(Boolean, default=False)
    
    # Relationship to user (after conversion)
    user = relationship("User", back_populates="waitlist_entry", uselist=False, foreign_keys="User.waitlist_id")

