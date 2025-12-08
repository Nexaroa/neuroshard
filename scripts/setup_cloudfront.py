#!/usr/bin/env python3
"""
CloudFront Setup for NeuroShard Training Data

This script:
1. Creates a CloudFront distribution for the S3 bucket
2. Configures Origin Access Control (OAC) so only CloudFront can access S3
3. Updates S3 bucket policy to allow CloudFront access
4. Optionally sets up a custom domain (data.neuroshard.com)

Security benefits:
- S3 bucket becomes private (no direct public access)
- CloudFront provides caching, DDoS protection, global edge locations
- Reduced S3 costs from caching
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
    print("WARNING: No .env file found")

load_env()

# AWS Clients
REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
BUCKET_NAME = 'neuroshard-training-data'

s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=REGION
)

cloudfront = boto3.client('cloudfront',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=REGION
)

def check_existing_distribution():
    """Check if a CloudFront distribution already exists for this bucket."""
    print("\n=== Checking for existing CloudFront distributions ===")
    
    paginator = cloudfront.get_paginator('list_distributions')
    for page in paginator.paginate():
        items = page.get('DistributionList', {}).get('Items', [])
        for dist in items:
            origins = dist.get('Origins', {}).get('Items', [])
            for origin in origins:
                if BUCKET_NAME in origin.get('DomainName', ''):
                    print(f"Found existing distribution: {dist['Id']}")
                    print(f"  Domain: {dist['DomainName']}")
                    print(f"  Status: {dist['Status']}")
                    return dist
    
    print("No existing distribution found.")
    return None

def create_origin_access_control():
    """Create an Origin Access Control for CloudFront -> S3."""
    print("\n=== Creating Origin Access Control ===")
    
    oac_name = f"neuroshard-{BUCKET_NAME}-oac"
    
    # Check if OAC already exists
    try:
        oacs = cloudfront.list_origin_access_controls()
        for oac in oacs.get('OriginAccessControlList', {}).get('Items', []):
            if oac['Name'] == oac_name:
                print(f"OAC already exists: {oac['Id']}")
                return oac['Id']
    except Exception as e:
        print(f"Error listing OACs: {e}")
    
    # Create new OAC
    try:
        response = cloudfront.create_origin_access_control(
            OriginAccessControlConfig={
                'Name': oac_name,
                'Description': 'OAC for NeuroShard training data bucket',
                'SigningProtocol': 'sigv4',
                'SigningBehavior': 'always',
                'OriginAccessControlOriginType': 's3'
            }
        )
        oac_id = response['OriginAccessControl']['Id']
        print(f"Created OAC: {oac_id}")
        return oac_id
    except Exception as e:
        print(f"Error creating OAC: {e}")
        return None

def create_cloudfront_distribution(oac_id):
    """Create CloudFront distribution."""
    print("\n=== Creating CloudFront Distribution ===")
    
    origin_id = f"S3-{BUCKET_NAME}"
    
    distribution_config = {
        'CallerReference': f'neuroshard-{int(time.time())}',
        'Comment': 'NeuroShard Training Data Distribution',
        'Enabled': True,
        'Origins': {
            'Quantity': 1,
            'Items': [{
                'Id': origin_id,
                'DomainName': f'{BUCKET_NAME}.s3.{REGION}.amazonaws.com',
                'S3OriginConfig': {
                    'OriginAccessIdentity': ''  # Empty for OAC
                },
                'OriginAccessControlId': oac_id,
            }]
        },
        'DefaultCacheBehavior': {
            'TargetOriginId': origin_id,
            'ViewerProtocolPolicy': 'redirect-to-https',
            'AllowedMethods': {
                'Quantity': 2,
                'Items': ['GET', 'HEAD'],
                'CachedMethods': {
                    'Quantity': 2,
                    'Items': ['GET', 'HEAD']
                }
            },
            'CachePolicyId': '658327ea-f89d-4fab-a63d-7e88639e58f6',  # CachingOptimized
            'Compress': True,
        },
        'PriceClass': 'PriceClass_100',  # US, Canada, Europe (cheapest)
        'HttpVersion': 'http2and3',
        'DefaultRootObject': 'manifest.json',
    }
    
    try:
        response = cloudfront.create_distribution(
            DistributionConfig=distribution_config
        )
        dist = response['Distribution']
        print(f"Created distribution: {dist['Id']}")
        print(f"Domain: {dist['DomainName']}")
        print(f"Status: {dist['Status']}")
        return dist
    except Exception as e:
        print(f"Error creating distribution: {e}")
        return None

def update_bucket_policy(distribution_arn):
    """Update S3 bucket policy to allow CloudFront access."""
    print("\n=== Updating S3 Bucket Policy ===")
    
    # Get AWS account ID from distribution ARN
    # ARN format: arn:aws:cloudfront::ACCOUNT_ID:distribution/DIST_ID
    account_id = distribution_arn.split(':')[4]
    
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
        print("Bucket policy updated successfully.")
        print(f"Only CloudFront distribution {distribution_arn} can access the bucket.")
    except Exception as e:
        print(f"Error updating bucket policy: {e}")

def disable_public_access():
    """Disable direct public access to S3 bucket."""
    print("\n=== Configuring S3 Public Access Block ===")
    
    try:
        s3.put_public_access_block(
            Bucket=BUCKET_NAME,
            PublicAccessBlockConfiguration={
                'BlockPublicAcls': True,
                'IgnorePublicAcls': True,
                'BlockPublicPolicy': False,  # Allow CloudFront policy
                'RestrictPublicBuckets': False  # Allow CloudFront access
            }
        )
        print("Public access block configured.")
    except Exception as e:
        print(f"Error configuring public access block: {e}")

def print_summary(dist):
    """Print setup summary."""
    print("\n" + "="*60)
    print("CLOUDFRONT SETUP COMPLETE")
    print("="*60)
    print(f"""
CloudFront Domain: https://{dist['DomainName']}

Update your code to use this URL:
  OLD: https://{BUCKET_NAME}.s3.amazonaws.com/
  NEW: https://{dist['DomainName']}/

Example:
  Manifest: https://{dist['DomainName']}/manifest.json
  Shard 0:  https://{dist['DomainName']}/shard_0.pt

Status: {dist['Status']}
(Distribution takes 5-15 minutes to deploy globally)

Security:
  ✓ S3 bucket is now private (no direct access)
  ✓ CloudFront provides caching and DDoS protection
  ✓ All traffic goes through CloudFront edge locations
""")

def main():
    print("="*60)
    print("NeuroShard CloudFront Setup")
    print("="*60)
    
    # Check for existing distribution
    existing = check_existing_distribution()
    if existing:
        print("\nDistribution already exists. Use AWS Console to manage it.")
        print(f"CloudFront URL: https://{existing['DomainName']}")
        return
    
    # Create OAC
    oac_id = create_origin_access_control()
    if not oac_id:
        print("Failed to create OAC. Aborting.")
        return
    
    # Create distribution
    dist = create_cloudfront_distribution(oac_id)
    if not dist:
        print("Failed to create distribution. Aborting.")
        return
    
    # Update bucket policy
    update_bucket_policy(dist['ARN'])
    
    # Configure public access block
    disable_public_access()
    
    # Print summary
    print_summary(dist)

if __name__ == "__main__":
    main()

