"use client"

import { useState, useCallback } from "react"
import { Upload, FileVideo, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

interface VideoUploadProps {
    onUpload: (file: File) => void
    isUploading: boolean
    uploadProgress: number
}

export function VideoUpload({ onUpload, isUploading, uploadProgress }: VideoUploadProps) {
    const [dragActive, setDragActive] = useState(false)
    const [selectedFile, setSelectedFile] = useState<File | null>(null)

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true)
        } else if (e.type === "dragleave") {
            setDragActive(false)
        }
    }, [])

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(false)
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setSelectedFile(e.dataTransfer.files[0])
        }
    }, [])

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        e.preventDefault()
        if (e.target.files && e.target.files[0]) {
            setSelectedFile(e.target.files[0])
        }
    }

    const handleUpload = () => {
        if (selectedFile) {
            onUpload(selectedFile)
        }
    }

    const clearFile = () => {
        setSelectedFile(null)
    }

    return (
        <Card className="w-full">
            <CardContent className="p-6">
                <div
                    className={cn(
                        "relative flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg transition-colors",
                        dragActive ? "border-primary bg-primary/5" : "border-muted-foreground/25",
                        isUploading && "opacity-50 pointer-events-none"
                    )}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                >
                    {!selectedFile ? (
                        <>
                            <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                <Upload className="w-12 h-12 mb-4 text-muted-foreground" />
                                <p className="mb-2 text-sm text-muted-foreground">
                                    <span className="font-semibold">Click to upload</span> or drag and drop
                                </p>
                                <p className="text-xs text-muted-foreground">MP4, WEBM, or MOV (MAX. 800MB)</p>
                            </div>
                            <input
                                id="dropzone-file"
                                type="file"
                                className="hidden"
                                accept="video/*"
                                onChange={handleChange}
                            />
                            <label
                                htmlFor="dropzone-file"
                                className="absolute inset-0 w-full h-full cursor-pointer"
                            />
                        </>
                    ) : (
                        <div className="flex flex-col items-center p-4">
                            <FileVideo className="w-16 h-16 text-primary mb-4" />
                            <p className="text-sm font-medium mb-2">{selectedFile.name}</p>
                            <p className="text-xs text-muted-foreground mb-4">
                                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                            </p>
                            {!isUploading && (
                                <Button variant="ghost" size="sm" onClick={clearFile} className="absolute top-2 right-2">
                                    <X className="w-4 h-4" />
                                </Button>
                            )}
                        </div>
                    )}
                </div>

                {selectedFile && (
                    <div className="mt-4 space-y-4">
                        {isUploading && (
                            <div className="space-y-2">
                                <div className="flex justify-between text-xs text-muted-foreground">
                                    <span>Uploading...</span>
                                    <span>{uploadProgress}%</span>
                                </div>
                                <Progress value={uploadProgress} />
                            </div>
                        )}
                        <Button
                            className="w-full"
                            onClick={handleUpload}
                            disabled={!selectedFile || isUploading}
                        >
                            {isUploading ? "Processing..." : "Analyze Video"}
                        </Button>
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
