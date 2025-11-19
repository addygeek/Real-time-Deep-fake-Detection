import { CheckCircle2, Clock, AlertTriangle } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

export interface QueueItem {
    id: string
    filename: string
    status: "processing" | "completed" | "flagged"
    timestamp: string
}

interface ProcessingQueueProps {
    items: QueueItem[]
}

export function ProcessingQueue({ items }: ProcessingQueueProps) {
    const getStatusIcon = (status: QueueItem["status"]) => {
        switch (status) {
            case "processing":
                return <Clock className="w-4 h-4 text-blue-500 animate-pulse" />
            case "completed":
                return <CheckCircle2 className="w-4 h-4 text-green-500" />
            case "flagged":
                return <AlertTriangle className="w-4 h-4 text-red-500" />
        }
    }

    const getStatusColor = (status: QueueItem["status"]) => {
        switch (status) {
            case "processing":
                return "secondary"
            case "completed":
                return "default" // Using default for green/success in this context if customized, or outline
            case "flagged":
                return "destructive"
        }
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle className="text-lg font-medium">Processing Queue</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="space-y-4">
                    {items.length === 0 ? (
                        <p className="text-sm text-muted-foreground text-center py-4">No items in queue</p>
                    ) : (
                        items.map((item) => (
                            <div
                                key={item.id}
                                className="flex items-center justify-between p-3 border rounded-lg bg-card/50"
                            >
                                <div className="flex items-center gap-3">
                                    {getStatusIcon(item.status)}
                                    <div>
                                        <p className="text-sm font-medium">{item.filename}</p>
                                        <p className="text-xs text-muted-foreground">{item.timestamp}</p>
                                    </div>
                                </div>
                                <Badge variant={getStatusColor(item.status)}>
                                    {item.status.charAt(0).toUpperCase() + item.status.slice(1)}
                                </Badge>
                            </div>
                        ))
                    )}
                </div>
            </CardContent>
        </Card>
    )
}
