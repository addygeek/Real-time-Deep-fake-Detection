"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis, Tooltip } from "recharts"

const data = [
    { name: "Mon", legitimate: 40, deepfake: 24 },
    { name: "Tue", legitimate: 30, deepfake: 13 },
    { name: "Wed", legitimate: 20, deepfake: 58 },
    { name: "Thu", legitimate: 27, deepfake: 39 },
    { name: "Fri", legitimate: 18, deepfake: 48 },
    { name: "Sat", legitimate: 23, deepfake: 38 },
    { name: "Sun", legitimate: 34, deepfake: 43 },
]

export function AnalyticsDashboard() {
    return (
        <Card className="col-span-4">
            <CardHeader>
                <CardTitle>Detection Analytics</CardTitle>
            </CardHeader>
            <CardContent className="pl-2">
                <div className="h-[200px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data}>
                            <XAxis
                                dataKey="name"
                                stroke="#888888"
                                fontSize={12}
                                tickLine={false}
                                axisLine={false}
                            />
                            <YAxis
                                stroke="#888888"
                                fontSize={12}
                                tickLine={false}
                                axisLine={false}
                                tickFormatter={(value) => `${value}`}
                            />
                            <Tooltip
                                contentStyle={{ background: '#1f2937', border: 'none', borderRadius: '8px' }}
                                itemStyle={{ color: '#fff' }}
                            />
                            <Bar
                                dataKey="legitimate"
                                fill="currentColor"
                                radius={[4, 4, 0, 0]}
                                className="fill-primary"
                            />
                            <Bar
                                dataKey="deepfake"
                                fill="currentColor"
                                radius={[4, 4, 0, 0]}
                                className="fill-destructive"
                            />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </CardContent>
        </Card>
    )
}
