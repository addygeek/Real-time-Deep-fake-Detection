const express = require('express');
const router = express.Router();
const blockchainController = require('../controllers/blockchainController');

router.post('/verify', blockchainController.verify);
router.post('/compare', blockchainController.compareVideos);
router.get('/stats', blockchainController.getChainStats);
router.get('/block/:hash', blockchainController.getBlockByHash);

module.exports = router;
