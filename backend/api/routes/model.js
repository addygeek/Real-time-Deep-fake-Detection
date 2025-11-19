const express = require('express');
const router = express.Router();
const modelController = require('../controllers/modelController');

router.post('/retrain', modelController.retrainModel);

module.exports = router;
