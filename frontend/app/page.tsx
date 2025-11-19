"use client"

import { useEffect } from "react"
import { Shield, Activity, Database, Lock } from "lucide-react"
import { VideoUpload } from "@/components/VideoUpload"
import { ProcessingQueue } from "@/components/ProcessingQueue"
import { ResultsCard } from "@/components/ResultsCard"
import { BlockchainStatus } from "@/components/BlockchainStatus"
import { AnalyticsDashboard } from "@/components/AnalyticsDashboard"
import { useApi } from "@/hooks/useApi"

export default function Home() {
    const {
        isUploading,
        uploadProgress,
        currentResult,
        queue,
        blockchainData,
        uploadVideo,
        loadInitialQueue
    } = useApi()

    useEffect(() => {
        loadInitialQueue()
    }, [loadInitialQueue])

    return (
        <main className="min-h-screen bg-background text-foreground p-8">
            <div className="max-w-7xl mx-auto space-y-8">
                {/* Header */}
                <div className="flex items-center justify-between border-b border-border pb-6">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-primary/10 rounded-lg">
                            <Shield className="w-8 h-8 text-primary" />
                        </div>
                        <div>
                            <h1 className="text-3xl font-bold tracking-tight">SpectraShield</h1>
                            <p className="text-muted-foreground">Multimodal Deepfake Detection System</p>
                        </div>
                    </div>
                    <div className="flex gap-4 text-sm text-muted-foreground">
                        <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4" />
                            <span>System Active</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <Database className="w-4 h-4" />
                            <span>v2.4.0 Model</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <Lock className="w-4 h-4" />
                            <span>Blockchain Connected</span>
                        </div>
                    </div>
                </div>

                {/* Main Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                    {/* Left Column: Upload & Queue */}
                    <div className="lg:col-span-4 space-y-6">
                        <section>
                            <h2 className="text-lg font-semibold mb-4">Input Stream</h2>
                            <VideoUpload
                                onUpload={uploadVideo}
                                isUploading={isUploading}
                                uploadProgress={uploadProgress}
                            />
                        </section>

                        <section>
                            <ProcessingQueue items={queue} />
                        </section>
                    </div>

                    {/* Right Column: Results & Analytics */}
                    <div className="lg:col-span-8 space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <section>
                                <h2 className="text-lg font-semibold mb-4">Live Analysis</h2>
                                <ResultsCard result={currentResult} />
                            </section>

                            <section>
                                <h2 className="text-lg font-semibold mb-4">Provenance</h2>
                                <div className="space-y-4">
                                    <BlockchainStatus
                                        hash={blockchainData?.hash || null}
                                        isVerified={!!blockchainData}
                                        timestamp={blockchainData?.timestamp || null}
                                    />
                                    {/* Placeholder for more provenance info if needed */}
                                    {!blockchainData && (
                                        <div className="h-32 border border-dashed rounded-lg flex items-center justify-center text-muted-foreground text-sm">
                                            Waiting for analysis...
                                        </div>
                                    )}
                                </div>
                            </section>
                        </div>

                        <section>
                            <h2 className="text-lg font-semibold mb-4">System Analytics</h2>
                            <AnalyticsDashboard />
                        </section>
                    </div>
                </div>
            </div>
        </main>
    )
}
