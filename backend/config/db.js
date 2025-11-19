const mongoose = require('mongoose');

const connectDB = async () => {
    try {
        // Use a mock connection string if not provided, or handle connection failure gracefully for demo
        const connStr = process.env.DB_URI || 'mongodb://localhost:27017/spectrashield';

        // For this demo environment where Mongo might not be running, we'll just log
        console.log(`Attempting to connect to MongoDB at ${connStr}...`);

        // await mongoose.connect(connStr);
        // console.log(`MongoDB Connected: ${mongoose.connection.host}`);

        console.log('MongoDB connection skipped for demo (uncomment in production)');
    } catch (error) {
        console.error(`Error: ${error.message}`);
        process.exit(1);
    }
};

module.exports = connectDB;
