const Analysis = require('../../models/Analysis');

exports.getSummary = async (req, res) => {
    try {
        const totalAnalyses = await Analysis.countDocuments();
        const completedAnalyses = await Analysis.countDocuments({ status: 'completed' });
        const fakeDetections = await Analysis.countDocuments({ 'result.isManipulated': true });

        const recentAnalyses = await Analysis.find()
            .sort({ createdAt: -1 })
            .limit(5)
            .select('filename status result.fakeProbability createdAt');

        res.json({
            success: true,
            summary: {
                total: totalAnalyses,
                completed: completedAnalyses,
                fakeCount: fakeDetections,
                realCount: completedAnalyses - fakeDetections
            },
            recent: recentAnalyses
        });

    } catch (error) {
        console.error('Analytics Summary Error:', error);
        res.status(500).json({ success: false, message: 'Server Error' });
    }
};
