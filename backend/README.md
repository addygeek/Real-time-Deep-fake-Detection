# SpectraShield Backend

Node.js/Express backend service for SpectraShield.

## Features
- **API**: REST endpoints for upload, analysis, and analytics.
- **Queue**: BullMQ (Redis) for asynchronous video processing.
- **Real-time**: Socket.io for live status updates.
- **Worker**: Integrated worker spawning Python inference scripts.

## Setup

1. **Install Dependencies**:
   ```bash
   npm install
   ```

2. **Environment Variables**:
   Ensure you have a `.env` file (see root `.env.example`).
   Required: `REDIS_HOST`, `REDIS_PORT`, `DB_URI`.

3. **Run**:
   ```bash
   # Development (with hot reload)
   npm run dev
   
   # Production
   npm start
   ```

## Architecture
- `api/`: Routes and Controllers.
- `core/`: Shared services (Queue, Socket).
- `workers/`: Background job processors.
- `models/`: Database schemas.
- `config/`: Configuration files.

## API Endpoints

### Upload
- `POST /upload`: Upload a video file (form-data key: `video`). Returns `analysisId`.

### Analysis
- `GET /analysis/status/:id`: Get the processing status of an analysis.
- `GET /analysis/results/:id`: Get the final results of a completed analysis.

### Blockchain
- `POST /blockchain/verify`: Verify an analysis result on the ledger. Body: `{ "analysisId": "..." }`.

### Analytics
- `GET /analytics/summary`: Get global stats (total processed, fake count, etc.).

### Model
- `POST /model/retrain`: Trigger the adaptive learning pipeline.

### System
- `GET /health`: Check system health (DB, Redis connection).

