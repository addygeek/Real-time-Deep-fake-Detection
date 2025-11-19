import { AlertTriangle, CheckCircle, Fingerprint, Activity } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

interface AnalysisResult {
    fakeProbability: number
    mismatchScore: number
    compressionSignature: string
    isManipulated: boolean
}

interface ResultsCardProps {
    result: AnalysisResult | null
}

export function ResultsCard({ result }: ResultsCardProps) {
    if (!result) {
        return (
            <Card className="h-full flex items-center justify-center min-h-[300px]">
                <CardContent className="text-center text-muted-foreground">
                    <Activity className="w-12 h-12 mx-auto mb-4 opacity-20" />
                    <p>Upload a video to see analysis results</p>
                </CardContent>
            </Card>
        )
    }

    const isHighRisk = result.fakeProbability > 70

    return (
        <Card className="h-full">
            <CardHeader>
                <CardTitle className="flex items-center justify-between">
                    Analysis Results
                    {result.isManipulated ? (
                        <span className="flex items-center gap-2 text-red-500 text-sm bg-red-500/10 px-3 py-1 rounded-full">
                            <AlertTriangle className="w-4 h-4" />
                            Potential Deepfake
                        </span>
                    ) : (
                        <span className="flex items-center gap-2 text-green-500 text-sm bg-green-500/10 px-3 py-1 rounded-full">
                            <CheckCircle className="w-4 h-4" />
                            Likely Authentic
                        </span>
                    )}
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
                <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Fake Probability</span>
                        <span className={isHighRisk ? "text-red-500 font-bold" : "text-green-500 font-bold"}>
                            {result.fakeProbability.toFixed(1)}%
                        </span>
                    </div>
                    <Progress value={result.fakeProbability} className={isHighRisk ? "bg-red-100" : "bg-green-100"} />
                </div>

                <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 rounded-lg bg-secondary/50 space-y-2">
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Activity className="w-4 h-4" />
                            A/V Mismatch
                        </div>
                        <p className="text-2xl font-semibold">{result.mismatchScore.toFixed(2)}</p>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50 space-y-2">
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Fingerprint className="w-4 h-4" />
                            Compression
                        </div>
                        <p className="text-xs font-mono break-all">{result.compressionSignature}</p>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
