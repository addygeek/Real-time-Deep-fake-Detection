require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const http = require('http');
const connectDB = require('./config/db');
const { initSocket } = require('./core/socket');

// Initialize App
const app = express();
const server = http.createServer(app);

// Initialize Socket.io
initSocket(server);

// Initialize Worker
require('./workers/inferenceWorker');

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('dev'));
app.use(express.json());

// Database Connection
connectDB();

// Routes
app.use('/health', require('./api/routes/health'));
app.use('/upload', require('./api/routes/upload'));
app.use('/analysis', require('./api/routes/analysis'));
app.use('/blockchain', require('./api/routes/blockchain'));
app.use('/analytics', require('./api/routes/analytics'));
app.use('/model', require('./api/routes/model'));

// Error Handling
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ success: false, message: 'Server Error' });
});

// Start Server
const PORT = process.env.PORT || 4000;
server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
