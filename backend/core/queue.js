const { Queue } = require('bullmq');
const connection = require('../config/redis');

const videoQueue = new Queue('video-processing', { connection });

const addJob = async (data) => {
    return await videoQueue.add('analyze-video', data);
};

module.exports = {
    videoQueue,
    addJob
};
