const Analysis = require('../../models/Analysis');

exports.getStatus = async (req, res) => {
    try {
        const analysis = await Analysis.findById(req.params.id).select('status');
        if (!analysis) {
            return res.status(404).json({ success: false, message: 'Analysis not found' });
        }
        res.json({ success: true, status: analysis.status });
    } catch (error) {
        console.error('Get Status Error:', error);
        res.status(500).json({ success: false, message: 'Server Error' });
    }
};

exports.getResult = async (req, res) => {
    try {
        const analysis = await Analysis.findById(req.params.id);
        if (!analysis) {
            return res.status(404).json({ success: false, message: 'Analysis not found' });
        }

        if (analysis.status !== 'completed') {
            return res.status(400).json({
                success: false,
                message: 'Analysis not yet completed',
                status: analysis.status
            });
        }

        res.json({ success: true, result: analysis.result });
    } catch (error) {
        console.error('Get Result Error:', error);
        res.status(500).json({ success: false, message: 'Server Error' });
    }
};
