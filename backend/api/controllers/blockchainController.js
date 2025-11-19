const Analysis = require('../../models/Analysis');
const { getBlockchain } = require('../../core/blockchain');

exports.verify = async (req, res) => {
    try {
        const { analysisId } = req.body;

        const analysis = await Analysis.findById(analysisId);
        if (!analysis) {
            return res.status(404).json({ success: false, message: 'Analysis not found' });
        }

        if (analysis.status !== 'completed') {
            return res.status(400).json({ success: false, message: 'Analysis not completed' });
        }

        const blockchain = getBlockchain();

        // Record analysis on blockchain if not already recorded
        if (!analysis.blockchainHash) {
            const blockchainData = await blockchain.recordAnalysis({
                analysisId: analysis._id.toString(),
                filePath: analysis.filename,
                filename: analysis.originalName,
                uploaderId: req.body.uploaderId || 'anonymous',
                result: analysis.result
            });

            analysis.blockchainHash = blockchainData.blockHash;
            analysis.videoHash = blockchainData.videoHash;
            await analysis.save();
        }

        // Verify the blockchain integrity
        const verification = await blockchain.verifyVideo(analysis.videoHash);

        res.json({
            success: true,
            verified: verification.verified,
            blockchainHash: analysis.blockchainHash,
            videoHash: analysis.videoHash,
            timestamp: verification.timestamp,
            uploaderId: verification.uploaderId,
            blockIndex: verification.blockIndex,
            message: "Analysis result verified on blockchain"
        });

    } catch (error) {
        console.error('Blockchain Verify Error:', error);
        res.status(500).json({ success: false, message: 'Server Error' });
    }
};

exports.compareVideos = async (req, res) => {
    try {
        const { originalHash, newVideoPath } = req.body;

        if (!originalHash) {
            return res.status(400).json({ success: false, message: 'Original hash required' });
        }

        const blockchain = getBlockchain();
        const comparison = await blockchain.compareVideos(originalHash, newVideoPath || 'new_upload');

        res.json({
            success: true,
            ...comparison
        });

    } catch (error) {
        console.error('Video Comparison Error:', error);
        res.status(500).json({ success: false, message: 'Server Error' });
    }
};

exports.getChainStats = async (req, res) => {
    try {
        const blockchain = getBlockchain();
        const stats = blockchain.getChainStats();

        res.json({
            success: true,
            stats: stats
        });

    } catch (error) {
        console.error('Chain Stats Error:', error);
        res.status(500).json({ success: false, message: 'Server Error' });
    }
};

exports.getBlockByHash = async (req, res) => {
    try {
        const { hash } = req.params;
        const blockchain = getBlockchain();
        const block = blockchain.findBlockByHash(hash);

        if (!block) {
            return res.status(404).json({ success: false, message: 'Block not found' });
        }

        res.json({
            success: true,
            block: block
        });

    } catch (error) {
        console.error('Get Block Error:', error);
        res.status(500).json({ success: false, message: 'Server Error' });
    }
};
