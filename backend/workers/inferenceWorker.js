const { Worker } = require('bullmq');
const connection = require('../config/redis');
const { getIo } = require('../core/socket');
const Analysis = require('../models/Analysis');
const axios = require('axios');

const ML_ENGINE_URL = process.env.ML_ENGINE_URL || 'http://localhost:5000';

const worker = new Worker('video-processing', async (job) => {
    const { analysisId, filePath } = job.data;
    const io = getIo();

    console.log(`Processing job ${job.id} for analysis ${analysisId}`);

    // Update status to processing
    await Analysis.findByIdAndUpdate(analysisId, { status: 'processing' });
    io.to(analysisId).emit('status-update', { status: 'processing' });

    try {
        // Call ML Engine API
        // Note: In Docker, filePath must be a shared volume path
        const response = await axios.post(`${ML_ENGINE_URL}/predict`, {
            filePath: filePath
        });

        const result = response.data;

        // Map Python result to our schema
        const analysisResult = {
            fakeProbability: result.confidence * 100,
            mismatchScore: result.artifacts.audio_mismatch,
            compressionSignature: "H.264/MPEG-4 AVC (High Profile)", // Mocked for now
            isManipulated: result.is_fake,
            details: result
        };

        // Update DB
        await Analysis.findByIdAndUpdate(analysisId, {
            status: 'completed',
            result: analysisResult
        });

        // Emit completion
        io.to(analysisId).emit('analysis-complete', analysisResult);
        console.log(`Job ${job.id} completed`);

    } catch (error) {
        console.error(`Job ${job.id} failed:`, error.message);
        await Analysis.findByIdAndUpdate(analysisId, { status: 'failed' });
        io.to(analysisId).emit('status-update', { status: 'failed', error: error.message });
        throw error;
    }

}, { connection });

module.exports = worker;
