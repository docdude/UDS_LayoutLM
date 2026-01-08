# GitHub Repository Setup Instructions

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `UDS_LayoutLM`
3. Description: `Extract HRSA UDS clinical quality metrics from EHR documents using LayoutLMv3`
4. **Keep it Private** (recommended for healthcare data)
5. **DO NOT** initialize with README (we have one)
6. Click "Create repository"

## Step 2: Initial Commit

```bash
# Make sure you're in the project directory
cd C:/Users/jloya/Documents/UDS_LayoutLM

# Add all files (respects .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: UDS LayoutLM for clinical document understanding

- Added LayoutLMv3-based extraction pipeline
- Complete Label Studio annotation setup
- 60+ UDS entity labels (colorectal, diabetes, hypertension, etc.)
- PDF processing with OCR
- Training and inference scripts
- Annotation validation tools
"

# Check status
git status
```

## Step 3: Push to GitHub

Replace `YOUR_USERNAME` with your GitHub username:

```bash
# Add remote (from GitHub's instructions)
git remote add origin https://github.com/YOUR_USERNAME/UDS_LayoutLM.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Verify Upload

1. Go to https://github.com/YOUR_USERNAME/UDS_LayoutLM
2. Verify files are there (README should display)
3. Check that `data/` folders are empty (PDFs not uploaded)

## Important: Protect PHI/PII

Your `.gitignore` already prevents committing:
- ❌ PDF files (patient data)
- ❌ Processed images
- ❌ Labeled annotations
- ❌ Trained models (large files)

✅ Only code, configs, and documentation are committed.

## Future Updates

```bash
# After making changes
git add .
git commit -m "Description of changes"
git push
```

## Collaboration

To add collaborators:
1. GitHub repo → Settings → Collaborators
2. Add team members by username

## Alternative: Use SSH

If you prefer SSH over HTTPS:

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub: Settings → SSH Keys → New SSH key
# Copy the public key:
cat ~/.ssh/id_ed25519.pub

# Use SSH remote instead
git remote set-url origin git@github.com:YOUR_USERNAME/UDS_LayoutLM.git
```

## Troubleshooting

### Large files error
If you accidentally try to commit large files:
```bash
git rm --cached data/raw_pdfs/*.pdf
git commit --amend
```

### Already have files staged?
```bash
git reset
git add .
git commit -m "Your message"
```
