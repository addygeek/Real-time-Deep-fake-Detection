# Git Commit Guide for SpectraShield

## ✅ What SHOULD Be Committed

### Source Code
- ✅ All `.js`, `.ts`, `.tsx` files
- ✅ All `.py` files
- ✅ All `.jsx`, `.css`, `.scss` files
- ✅ Configuration files (non-secret)

### Documentation
- ✅ All `.md` files
- ✅ README files
- ✅ API documentation
- ✅ Architecture diagrams

### Configuration (Public)
- ✅ `package.json`
- ✅ `tsconfig.json`
- ✅ `next.config.js`
- ✅ `tailwind.config.ts`
- ✅ `.env.example` (template only)
- ✅ `docker-compose.yaml`
- ✅ Kubernetes manifests (non-secret)
- ✅ Terraform configs (non-secret)

### Project Structure
- ✅ Directory structure
- ✅ `.gitkeep` files
- ✅ `.gitignore`

### Small Assets
- ✅ Icons, logos (< 1MB)
- ✅ Sample images
- ✅ Fonts

### ML Models (Optional)
- ✅ Model architecture code
- ✅ Training scripts
- ⚠️ Small model weights (< 100MB)
- ⚠️ Use Git LFS for large models

---

## ❌ What SHOULD NOT Be Committed

### Dependencies
- ❌ `node_modules/`
- ❌ `__pycache__/`
- ❌ `.venv/`, `venv/`
- ❌ `package-lock.json` (optional)
- ❌ `yarn.lock` (optional)

### Build Output
- ❌ `.next/`
- ❌ `build/`, `dist/`
- ❌ `out/`
- ❌ Compiled files

### Environment & Secrets
- ❌ `.env` (actual secrets)
- ❌ `.env.local`
- ❌ `*.pem`, `*.key`
- ❌ API keys
- ❌ Database credentials
- ❌ Blockchain private keys

### User Data
- ❌ Uploaded videos
- ❌ User uploads
- ❌ Generated files
- ❌ Cache files

### Logs & Temp Files
- ❌ `*.log`
- ❌ `tmp/`, `temp/`
- ❌ `.cache/`
- ❌ Debug files

### OS & IDE Files
- ❌ `.DS_Store`
- ❌ `Thumbs.db`
- ❌ `.vscode/` (unless shared)
- ❌ `.idea/`

### Large Files
- ❌ Videos (`.mp4`, `.avi`, etc.)
- ❌ Large datasets
- ❌ Database dumps
- ❌ Large model files (> 100MB)

### Database Files
- ❌ `*.db`, `*.sqlite`
- ❌ `blockchain.json` (runtime data)
- ❌ Session data

---

## 📦 Git LFS (Large File Storage)

For large files that MUST be versioned:

```bash
# Install Git LFS
git lfs install

# Track large model files
git lfs track "*.pth"
git lfs track "*.h5"
git lfs track "ml-engine/models/*.pth"

# Commit .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"
```

---

## 🔒 Handling Secrets

### Use Environment Variables
```bash
# ❌ Don't commit
DB_PASSWORD=mysecretpassword

# ✅ Do commit (.env.example)
DB_PASSWORD=your_password_here
```

### Use Secret Management
- AWS Secrets Manager
- Azure Key Vault
- HashiCorp Vault
- Kubernetes Secrets

---

## 📝 Commit Best Practices

### Good Commit Messages
```bash
# ✅ Good
git commit -m "feat: Add blockchain verification endpoint"
git commit -m "fix: Resolve video upload timeout issue"
git commit -m "docs: Update API documentation"

# ❌ Bad
git commit -m "update"
git commit -m "fix stuff"
git commit -m "changes"
```

### Commit Message Format
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

---

## 🚀 Before First Commit

### 1. Clean Up
```bash
# Remove node_modules if accidentally added
git rm -r --cached node_modules
git rm -r --cached frontend/node_modules
git rm -r --cached backend/node_modules

# Remove .next build files
git rm -r --cached frontend/.next

# Remove Python cache
git rm -r --cached **/__pycache__
```

### 2. Add .gitignore
```bash
git add .gitignore
git commit -m "chore: Add comprehensive .gitignore"
```

### 3. Verify What Will Be Committed
```bash
# Check status
git status

# See what will be added
git add --dry-run .

# Review changes
git diff --cached
```

### 4. Commit in Logical Groups
```bash
# Backend
git add backend/
git commit -m "feat: Add complete backend implementation"

# Frontend
git add frontend/
git commit -m "feat: Add Next.js frontend with all components"

# ML Engine
git add ml-engine/
git commit -m "feat: Add ML engine with trained models"

# Documentation
git add *.md
git commit -m "docs: Add comprehensive documentation"
```

---

## 🔍 Verify Before Push

```bash
# Check what will be pushed
git log origin/main..HEAD

# Check file sizes
git ls-files | xargs ls -lh | sort -k5 -h -r | head -20

# Find large files
find . -size +10M -not -path "*/node_modules/*"

# Check for secrets
git secrets --scan
```

---

## 📊 Repository Size Management

### Check Repository Size
```bash
git count-objects -vH
```

### Remove Large Files from History
```bash
# Use BFG Repo-Cleaner
bfg --strip-blobs-bigger-than 100M

# Or git filter-branch
git filter-branch --tree-filter 'rm -f large-file.zip' HEAD
```

---

## ✅ Recommended Workflow

```bash
# 1. Stage changes
git add .

# 2. Check what's staged
git status

# 3. Review changes
git diff --cached

# 4. Commit with good message
git commit -m "feat: Add feature description"

# 5. Push to remote
git push origin main
```

---

## 🎯 Quick Reference

### Safe to Commit
```
✅ Source code (.js, .py, .ts, .tsx)
✅ Documentation (.md)
✅ Config files (public)
✅ Small assets (< 1MB)
✅ .gitignore, .gitkeep
```

### Never Commit
```
❌ node_modules/
❌ .env (with secrets)
❌ Build output (.next/, dist/)
❌ Uploaded files
❌ Large files (> 100MB)
❌ Secrets, keys, passwords
```

---

## 🆘 Emergency: Committed Secrets

```bash
# 1. Remove from latest commit
git reset HEAD~1
git add .gitignore
git commit -m "chore: Add .gitignore"

# 2. Remove from history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# 3. Force push (DANGER!)
git push origin --force --all

# 4. Rotate all exposed secrets immediately!
```

---

**Remember**: When in doubt, don't commit it! You can always add files later, but removing them from history is difficult.
