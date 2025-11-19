import { Shield, ExternalLink, Copy } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

interface BlockchainStatusProps {
    hash: string | null
    isVerified: boolean
    timestamp: string | null
}

export function BlockchainStatus({ hash, isVerified, timestamp }: BlockchainStatusProps) {
    if (!hash) return null

    return (
        <Card className="bg-slate-950 border-slate-800">
            <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-slate-400 flex items-center gap-2">
                    <Shield className="w-4 h-4 text-purple-500" />
                    Blockchain Provenance
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <div className="space-y-1">
                            <p className="text-xs text-slate-500">Transaction Hash</p>
                            <div className="flex items-center gap-2">
                                <code className="text-xs bg-slate-900 px-2 py-1 rounded text-purple-400 font-mono">
                                    {hash.slice(0, 6)}...{hash.slice(-6)}
                                </code>
                                <Button variant="ghost" size="icon" className="h-6 w-6 text-slate-500 hover:text-slate-300">
                                    <Copy className="w-3 h-3" />
                                </Button>
                            </div>
                        </div>
                        <div className="text-right space-y-1">
                            <p className="text-xs text-slate-500">Status</p>
                            <span className="text-xs text-green-500 flex items-center justify-end gap-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                                Verified
                            </span>
                        </div>
                    </div>

                    {timestamp && (
                        <div className="pt-2 border-t border-slate-800 flex justify-between items-center">
                            <p className="text-xs text-slate-500">Timestamp: {timestamp}</p>
                            <Button variant="link" size="sm" className="h-auto p-0 text-xs text-purple-500">
                                View on Explorer <ExternalLink className="w-3 h-3 ml-1" />
                            </Button>
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
    )
}
