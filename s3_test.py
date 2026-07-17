import boto3
import os
from dotenv import load_dotenv

load_dotenv()

# CONNECT TO S3
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')

# FUNCTION 1 - List all files in bucket
def list_files():
    print("\n📁 Files in S3 bucket:")
    response = s3.list_objects_v2(Bucket=BUCKET_NAME)
    
    if 'Contents' not in response:
        print("  Bucket is empty")
        return
    
    for obj in response['Contents']:
        print(f"  → {obj['Key']} ({obj['Size']} bytes)")

# FUNCTION 2 - Upload a file
def upload_file(local_path, s3_key):
    print(f"\n⬆️ Uploading {local_path} to S3...")
    s3.upload_file(local_path, BUCKET_NAME, s3_key)
    print(f"✅ Uploaded successfully as {s3_key}")

# FUNCTION 3 - Download a file
def download_file(s3_key, local_path):
    print(f"\n⬇️ Downloading {s3_key} from S3...")
    s3.download_file(BUCKET_NAME, s3_key, local_path)
    print(f"✅ Downloaded to {local_path}")

# FUNCTION 4 - Read file content directly
def read_file_content(s3_key):
    print(f"\n📖 Reading {s3_key} from S3...")
    response = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
    content = response['Body'].read()
    print(f"✅ Read {len(content)} bytes")
    return content

# TEST ALL FUNCTIONS
if __name__ == "__main__":
    print("🔗 Connecting to AWS S3...")
    
    # Test 1 - list files
    list_files()
    
    # Test 2 - upload a file
    # Create a test file first
    with open("test_document.txt", "w") as f:
        f.write("Hello from Python! This is my first S3 upload.")
    
    upload_file("test_document.txt", "documents/test_document.txt")
    
    # Test 3 - list again to see uploaded file
    list_files()
    
    # Test 4 - read file directly from S3
    content = read_file_content("documents/test_document.txt")
    print(f"\nFile content: {content.decode('utf-8')}") 