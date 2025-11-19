# SpectraShield Frontend

Next.js-based web interface for the SpectraShield deepfake detection platform.

## Features

- **Video Upload**: Drag-and-drop interface with progress tracking
- **Real-time Updates**: WebSocket integration for live analysis status
- **Results Visualization**: Interactive charts and confidence scores
- **Blockchain Verification**: Provenance tracking and verification
- **Analytics Dashboard**: System-wide statistics and trends
- **Processing Queue**: View all ongoing and completed analyses

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Custom components with Radix UI primitives
- **State Management**: React Hooks
- **API Client**: Fetch API
- **Real-time**: Socket.io-client

## Installation

```bash
npm install
```

## Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Build

```bash
npm run build
npm start
```

## Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:4000
```

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx       # Root layout
│   ├── page.tsx         # Home page
│   └── globals.css      # Global styles
├── components/
│   ├── VideoUpload.tsx
│   ├── ResultsCard.tsx
│   ├── ProcessingQueue.tsx
│   ├── BlockchainStatus.tsx
│   ├── AnalyticsDashboard.tsx
│   └── ui/              # Reusable UI components
├── services/
│   └── api.ts           # API client
├── hooks/
│   └── useApi.ts        # Custom hooks
└── lib/
    └── utils.ts         # Utility functions
```

## Components

### VideoUpload
Handles video file selection and upload with drag-and-drop support.

### ResultsCard
Displays analysis results with confidence scores and artifact detection.

### BlockchainStatus
Shows blockchain verification status and hash information.

### ProcessingQueue
Lists all videos in the processing queue with their status.

### AnalyticsDashboard
System-wide statistics and analytics visualization.

## API Integration

The frontend communicates with the backend API through the `api.ts` service:

- `uploadVideo()` - Upload video for analysis
- `getStatus()` - Poll for analysis status
- `getResults()` - Fetch analysis results
- `getBlockchainHash()` - Verify on blockchain
- `getQueue()` - Get processing queue

## Styling

Uses Tailwind CSS with a custom design system:

- Dark mode support
- Responsive design
- Custom color palette
- Smooth animations

## Deployment

### Vercel (Recommended)
```bash
vercel deploy
```

### Docker
```bash
docker build -t spectrashield-frontend .
docker run -p 3000:3000 spectrashield-frontend
```

## License

MIT
