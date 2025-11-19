const validateVideoUpload = (req, res, next) => {
    if (!req.file) {
        return res.status(400).json({ success: false, message: 'No video file provided' });
    }

    const allowedMimeTypes = ['video/mp4', 'video/webm', 'video/quicktime', 'video/x-msvideo'];
    if (!allowedMimeTypes.includes(req.file.mimetype)) {
        return res.status(400).json({
            success: false,
            message: 'Invalid file type. Only MP4, WEBM, MOV, and AVI are allowed'
        });
    }

    const maxSize = 50 * 1024 * 1024; // 50MB
    if (req.file.size > maxSize) {
        return res.status(400).json({
            success: false,
            message: 'File too large. Maximum size is 50MB'
        });
    }

    next();
};

const validateAnalysisId = (req, res, next) => {
    const { id } = req.params;

    if (!id || !id.match(/^[0-9a-fA-F]{24}$/)) {
        return res.status(400).json({
            success: false,
            message: 'Invalid analysis ID format'
        });
    }

    next();
};

const validateBlockchainVerify = (req, res, next) => {
    const { analysisId } = req.body;

    if (!analysisId) {
        return res.status(400).json({
            success: false,
            message: 'Analysis ID is required'
        });
    }

    if (!analysisId.match(/^[0-9a-fA-F]{24}$/)) {
        return res.status(400).json({
            success: false,
            message: 'Invalid analysis ID format'
        });
    }

    next();
};

module.exports = {
    validateVideoUpload,
    validateAnalysisId,
    validateBlockchainVerify
};
