"use client"

import { useState, useCallback } from "react"
import { api, AnalysisResult } from "@/services/api"
import { QueueItem } from "@/components/ProcessingQueue"

export function useApi() {
    const [isUploading, setIsUploading] = useState(false)
    const [uploadProgress, setUploadProgress] = useState(0)
    const [currentResult, setCurrentResult] = useState<AnalysisResult | null>(null)
    const [queue, setQueue] = useState<QueueItem[]>([])
    const [blockchainData, setBlockchainData] = useState<{ hash: string; timestamp: string } | null>(null)

    const uploadVideo = useCallback(async (file: File) => {
        setIsUploading(true)
        setUploadProgress(0)
        setCurrentResult(null)
        setBlockchainData(null)

        try {
            // Simulate upload progress
            const progressInterval = setInterval(() => {
                setUploadProgress((prev) => {
                    if (prev >= 90) {
                        clearInterval(progressInterval)
                        return 90
                    }
                    return prev + 10
                })
            }, 200)

            const { id } = await api.uploadVideo(file)

            clearInterval(progressInterval)
            setUploadProgress(100)

            // Add to queue
            const newItem: QueueItem = {
                id,
                filename: file.name,
                status: "processing",
                timestamp: "Just now"
            }
            setQueue((prev) => [newItem, ...prev])

            // Poll for results
            let attempts = 0;
            const maxAttempts = 60; // 2 minutes timeout

            const pollInterval = setInterval(async () => {
                try {
                    const status = await api.getStatus(id);

                    if (status === 'completed') {
                        clearInterval(pollInterval);
                        const result = await api.getResults(id);
                        setCurrentResult(result);

                        // Update queue item status
                        setQueue((prev) => prev.map(item =>
                            item.id === id
                                ? { ...item, status: result.isManipulated ? "flagged" : "completed" }
                                : item
                        ));

                        // Get blockchain data
                        const bcData = await api.getBlockchainHash(id);
                        setBlockchainData(bcData);
                        setIsUploading(false);
                    } else if (status === 'failed') {
                        clearInterval(pollInterval);
                        console.error("Analysis failed");
                        setIsUploading(false);
                    } else {
                        attempts++;
                        if (attempts >= maxAttempts) {
                            clearInterval(pollInterval);
                            console.error("Analysis timeout");
                            setIsUploading(false);
                        }
                    }
                } catch (e) {
                    console.error("Polling error", e);
                }
            }, 2000);

        } catch (error) {
            console.error("Upload failed:", error)
            setIsUploading(false)
        }
    }, [])

    const loadInitialQueue = useCallback(async () => {
        const items = await api.getQueue()
        setQueue(items)
    }, [])

    return {
        isUploading,
        uploadProgress,
        currentResult,
        queue,
        blockchainData,
        uploadVideo,
        loadInitialQueue
    }
}
