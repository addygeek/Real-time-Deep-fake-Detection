#!/usr/bin/env node

const { getBlockchain } = require('../core/blockchain');

async function initBlockchain() {
    console.log('Initializing SpectraShield Blockchain...');

    const blockchain = getBlockchain();
    const stats = blockchain.getChainStats();

    console.log('\n=== Blockchain Initialized ===');
    console.log(`Chain Length: ${stats.length}`);
    console.log(`Is Valid: ${stats.isValid}`);
    console.log(`Latest Block: ${stats.latestBlock}`);
    console.log(`Total Analyses: ${stats.totalAnalyses}`);
    console.log('\n✅ Blockchain ready for use');
}

initBlockchain().catch(console.error);
