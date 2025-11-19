const express = require('express');
const router = express.Router();
const analysisController = require('../controllers/analysisController');

router.get('/status/:id', analysisController.getStatus);
router.get('/results/:id', analysisController.getResult);

module.exports = router;
