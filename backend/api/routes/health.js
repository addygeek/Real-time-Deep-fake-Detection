const express = require('express');
const router = express.Router();
const mongoose = require('mongoose');
const redisConnection = require('../../config/redis');

router.get('/', async (req, res) => {
    const healthcheck = {
        uptime: process.uptime(),
        timestamp: Date.now(),
        message: 'OK',
        services: {
            database: 'disconnected',
            redis: 'disconnected'
        }
    };

    try {
        // Check MongoDB
        if (mongoose.connection.readyState === 1) {
            healthcheck.services.database = 'connected';
        } else {
            healthcheck.services.database = mongoose.connection.readyState;
        }

        // Check Redis
        if (redisConnection.status === 'ready') {
            healthcheck.services.redis = 'connected';
        } else {
            healthcheck.services.redis = redisConnection.status;
        }

        res.json(healthcheck);
    } catch (error) {
        healthcheck.message = error;
        res.status(503).json(healthcheck);
    }
});

module.exports = router;
