const mongoose = require('mongoose');

const AnalysisSchema = new mongoose.Schema({
    filename: {
        type: String,
        required: true
    },
    originalName: {
        type: String,
        required: true
    },
    status: {
        type: String,
        enum: ['queued', 'processing', 'completed', 'failed'],
        default: 'queued'
    },
    result: {
        fakeProbability: Number,
        mismatchScore: Number,
        compressionSignature: String,
        isManipulated: Boolean,
        details: Object
    },
    blockchainHash: String,
    videoHash: String,
    createdAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('Analysis', AnalysisSchema);
