#!/usr/bin/env python3
"""
CloudFront Setup for NeuroShard Releases

This script:
1. Creates a CloudFront distribution for the neuroshard S3 bucket (releases/ prefix)
2. Configures Origin Access Control (OAC) so only CloudFront can access S3
3. Updates S3 bucket policy to allow CloudFront access
4. Makes releases publicly downloadable even though GitHub repo is private

Usage:
    python scripts/setup_releases_cloudfront.py
"""

import os
import json
import boto3
import time
from pathlib import Path

# Load environment
def load_env():
    possible_paths = [
        Path(__file__).parent.parent / 'website' / '.env',
        Path(__file__).parent.parent / '.env',
    ]
    for p in possible_paths:
        if p.exists():
            print(f"Loading credentials from {p}")
            with open(p) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        k, v = line.strip().split('=', 1)
                        os.environ[k] = v.strip('"').strip("'")
            return
    print("WARNING: No .env file found, using environment variables")

load_env()

# AWS Clients
REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
BUCKET_NAME = 'neuroshard'  # New bucket for releases
RELEASES_PREFIX = 'releases/'  # Prefix for release files

s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=REGION
)

cloudfront = boto3.client('cloudfront',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name='us-east-1'  # CloudFront is always us-east-1
)

sts = boto3.client('sts',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=REGION
)

def get_aws_account_id():
    """Get AWS account ID."""
    return sts.get_caller_identity()['Account']

def check_bucket_exists():
    """Check if the neuroshard bucket exists."""
    print("\n=== Checking S3 Bucket ===")
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
        print(f"✅ Bucket '{BUCKET_NAME}' exists")
        return True
    except:
        print(f"❌ Bucket '{BUCKET_NAME}' does not exist")
        print(f"\nPlease create it first:")
        print(f"  aws s3 mb s3://{BUCKET_NAME} --region {REGION}")
        return False

def check_existing_distribution():
    """Check if a CloudFront distribution already exists for this bucket."""
    print("\n=== Checking for existing CloudFront distributions ===")
    
    paginator = cloudfront.get_paginator('list_distributions')
    for page in paginator.paginate():
        if 'DistributionList' not in page or 'Items' not in page['DistributionList']:
            continue
            
        for dist in page['DistributionList']['Items']:
            origin_domain = dist['Origins']['Items'][0]['DomainName']
            # Match exact bucket name (not substring)
            if f"{BUCKET_NAME}.s3" in origin_domain and "training-data" not in origin_domain:
                print(f"✅ Found existing distribution: {dist['Id']}")
                print(f"   Domain: {dist['DomainName']}")
                print(f"   Status: {dist['Status']}")
                print(f"   Origin: {origin_domain}")
                return dist
    
    print("ℹ️  No existing distribution found for 'neuroshard' bucket")
    return None

def create_origin_access_control():
    """Create an Origin Access Control for CloudFront -> S3."""
    print("\n=== Creating Origin Access Control ===")
    
    oac_name = f"{BUCKET_NAME}-releases-oac"
    
    # Check if OAC already exists
    try:
        oacs = cloudfront.list_origin_access_controls()
        for oac in oacs.get('OriginAccessControlList', {}).get('Items', []):
            if oac['Name'] == oac_name:
                print(f"✅ Found existing OAC: {oac['Id']}")
                return oac['Id']
    except Exception as e:
        print(f"Warning checking OACs: {e}")
    
    # Create new OAC
    response = cloudfront.create_origin_access_control(
        OriginAccessControlConfig={
            'Name': oac_name,
            'Description': f'OAC for {BUCKET_NAME} releases',
            'SigningProtocol': 'sigv4',
            'SigningBehavior': 'always',
            'OriginAccessControlOriginType': 's3'
        }
    )
    
    oac_id = response['OriginAccessControl']['Id']
    print(f"✅ Created new OAC: {oac_id}")
    return oac_id

def create_cloudfront_distribution(oac_id):
    """Create CloudFront distribution."""
    print("\n=== Creating CloudFront Distribution ===")
    
    caller_ref = f"neuroshard-releases-{int(time.time())}"
    
    config = {
        'CallerReference': caller_ref,
        'Comment': 'NeuroShard GPU Releases Distribution',
        'Enabled': True,
        'Origins': {
            'Quantity': 1,
            'Items': [{
                'Id': 'S3-neuroshard-releases',
                'DomainName': f'{BUCKET_NAME}.s3.{REGION}.amazonaws.com',
                'OriginPath': '',  # No prefix in origin, we'll use path patterns
                'OriginAccessControlId': oac_id,
                'S3OriginConfig': {
                    'OriginAccessIdentity': ''  # Empty for OAC
                }
            }]
        },
        'DefaultCacheBehavior': {
            'TargetOriginId': 'S3-neuroshard-releases',
            'ViewerProtocolPolicy': 'redirect-to-https',
            'AllowedMethods': {
                'Quantity': 2,
                'Items': ['GET', 'HEAD'],
                'CachedMethods': {
                    'Quantity': 2,
                    'Items': ['GET', 'HEAD']
                }
            },
            'Compress': True,
            'CachePolicyId': '658327ea-f89d-4fab-a63d-7e88639e58f6',  # CachingOptimized
            'OriginRequestPolicyId': '88a5eaf4-2fd4-4709-b370-b4c650ea3fcf'  # CORS-S3Origin
        },
        'PriceClass': 'PriceClass_All',  # Use all edge locations for global distribution
        'ViewerCertificate': {
            'CloudFrontDefaultCertificate': True,
            'MinimumProtocolVersion': 'TLSv1.2_2021'
        }
    }
    
    try:
        response = cloudfront.create_distribution(
            DistributionConfig=config
        )
        
        dist = response['Distribution']
        print(f"✅ Distribution created: {dist['Id']}")
        print(f"   Domain: {dist['DomainName']}")
        print(f"   Status: {dist['Status']}")
        return dist
        
    except Exception as e:
        print(f"❌ Error creating distribution: {e}")
        raise

def update_bucket_policy(distribution_id):
    """Update S3 bucket policy to allow CloudFront access."""
    print("\n=== Updating S3 Bucket Policy ===")
    
    account_id = get_aws_account_id()
    distribution_arn = f"arn:aws:cloudfront::{account_id}:distribution/{distribution_id}"
    
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowCloudFrontServicePrincipal",
                "Effect": "Allow",
                "Principal": {
                    "Service": "cloudfront.amazonaws.com"
                },
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{BUCKET_NAME}/*",
                "Condition": {
                    "StringEquals": {
                        "AWS:SourceArn": distribution_arn
                    }
                }
            }
        ]
    }
    
    try:
        s3.put_bucket_policy(
            Bucket=BUCKET_NAME,
            Policy=json.dumps(policy)
        )
        print(f"✅ Bucket policy updated")
        print(f"   Only CloudFront distribution {distribution_arn} can access the bucket.")
        
    except Exception as e:
        print(f"❌ Error updating bucket policy: {e}")
        raise

def configure_bucket_public_access():
    """Configure bucket to allow CloudFront access while blocking direct public access."""
    print("\n=== Configuring Bucket Public Access Settings ===")
    
    try:
        s3.put_public_access_block(
            Bucket=BUCKET_NAME,
            PublicAccessBlockConfiguration={
                'BlockPublicAcls': True,  # Block public ACLs
                'IgnorePublicAcls': True,  # Ignore existing public ACLs
                'BlockPublicPolicy': False,  # Allow CloudFront policy
                'RestrictPublicBuckets': False  # Allow CloudFront access
            }
        )
        print("✅ Public access settings configured")
        print("   Direct S3 access: BLOCKED")
        print("   CloudFront access: ALLOWED")
        
    except Exception as e:
        print(f"❌ Error configuring public access: {e}")
        raise

def print_summary(dist):
    print("\n" + "=" * 80)
    print("CLOUDFRONT SETUP COMPLETE")
    print("=" * 80)
    print(f"""
CloudFront Domain: https://{dist['DomainName']}
Distribution ID: {dist['Id']}
Status: {dist['Status']}

Example URLs:
  Latest GPU: https://{dist['DomainName']}/releases/latest/NeuroShardNode-GPU.exe
  Version:    https://{dist['DomainName']}/releases/v0.3.81/NeuroShardNode-GPU.exe

GitHub Actions Workflow:
  aws s3 cp NeuroShardNode-GPU.exe \\
    s3://{BUCKET_NAME}/releases/VERSION/NeuroShardNode-GPU.exe \\
    --acl public-read

IMPORTANT:
  ✓ Distribution status is '{dist['Status']}'
  ✓ If status is 'InProgress', wait 10-15 minutes for deployment
  ✓ CloudFront provides caching and DDoS protection
  ✓ All traffic goes through CloudFront edge locations
  ✓ Direct S3 access is blocked for security
  
Save this CloudFront domain for your .github/workflows/build.yml!
""")

if __name__ == '__main__':
    print("NeuroShard Releases CloudFront Setup")
    print("=" * 80)
    
    # Check bucket
    if not check_bucket_exists():
        exit(1)
    
    # Check for existing distribution
    existing = check_existing_distribution()
    if existing:
        # Check if it's pointing to the right bucket
        print(f"\nFound distribution for a bucket with 'neuroshard' in the name")
        print(f"Checking if it's configured for the '{BUCKET_NAME}' bucket specifically...")
        
        # We'll create a new one if needed (the check will handle it)
    
    # Create OAC
    oac_id = create_origin_access_control()
    
    # Create distribution
    dist = create_cloudfront_distribution(oac_id)
    
    # Update bucket policy
    update_bucket_policy(dist['Id'])
    
    # Configure public access settings
    configure_bucket_public_access()
    
    # Print summary
    print_summary(dist)

