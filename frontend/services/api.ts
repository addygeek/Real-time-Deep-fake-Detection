import { QueueItem } from "@/components/ProcessingQueue"

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000';

export interface AnalysisResult {
    fakeProbability: number
    mismatchScore: number
    compressionSignature: string
    isManipulated: boolean
    details?: any
}

export const api = {
    uploadVideo: async (file: File): Promise<{ id: string }> => {
        const formData = new FormData();
        formData.append('video', file);

        const res = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) throw new Error('Upload failed');
        const data = await res.json();
        return { id: data.analysisId };
    },

    getResults: async (id: string): Promise<AnalysisResult> => {
        const res = await fetch(`${API_URL}/analysis/results/${id}`);
        if (!res.ok) throw new Error('Failed to fetch results');
        const data = await res.json();
        return data.result;
    },

    getStatus: async (id: string): Promise<string> => {
        const res = await fetch(`${API_URL}/analysis/status/${id}`);
        if (!res.ok) throw new Error('Failed to fetch status');
        const data = await res.json();
        return data.status;
    },

    getQueue: async (): Promise<QueueItem[]> => {
        // For now, we'll hit the analytics endpoint or a specific queue endpoint if we had one.
        // Using analytics summary recent items as a proxy for queue/recent activity
        const res = await fetch(`${API_URL}/analytics/summary`);
        if (!res.ok) return [];
        const data = await res.json();

        return data.recent.map((item: any) => ({
            id: item._id,
            filename: item.filename,
            status: item.status,
            timestamp: new Date(item.createdAt).toLocaleTimeString()
        }));
    },

    getBlockchainHash: async (id: string): Promise<{ hash: string; timestamp: string }> => {
        const res = await fetch(`${API_URL}/blockchain/verify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ analysisId: id })
        });

        if (!res.ok) throw new Error('Verification failed');
        const data = await res.json();
        return {
            hash: data.blockchainHash,
            timestamp: data.timestamp
        };
    }
}
