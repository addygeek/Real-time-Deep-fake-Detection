const express = require('express');
const router = express.Router();
const uploadController = require('../controllers/uploadController');
const uploadMiddleware = require('../middlewares/uploadMiddleware');

router.post('/', uploadMiddleware.single('video'), uploadController.uploadVideo);

module.exports = router;
