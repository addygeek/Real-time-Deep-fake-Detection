const Analysis = require('../../models/Analysis');
const videoQueue = require('../../jobs/videoQueue');

exports.uploadVideo = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ success: false, message: 'No video file uploaded' });
        }

        // Create database record
        const analysis = await Analysis.create({
            filename: req.file.filename,
            originalName: req.file.originalname,
            status: 'queued'
        });

        // Add to job queue
        await videoQueue.add('analyze-video', {
            analysisId: analysis._id,
            filePath: req.file.path
        });

        res.status(201).json({
            success: true,
            message: 'Video uploaded and queued for analysis',
            analysisId: analysis._id
        });

    } catch (error) {
        console.error('Upload Error:', error);
        res.status(500).json({ success: false, message: 'Server Error' });
    }
};
