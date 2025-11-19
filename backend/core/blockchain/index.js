const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');

class Block {
    constructor(index, timestamp, data, previousHash = '') {
        this.index = index;
        this.timestamp = timestamp;
        this.data = data;
        this.previousHash = previousHash;
        this.hash = this.calculateHash();
        this.nonce = 0;
    }

    calculateHash() {
        return crypto.createHash('sha256')
            .update(this.index + this.previousHash + this.timestamp + JSON.stringify(this.data) + this.nonce)
            .digest('hex');
    }

    mineBlock(difficulty) {
        while (this.hash.substring(0, difficulty) !== Array(difficulty + 1).join("0")) {
            this.nonce++;
            this.hash = this.calculateHash();
        }
    }
}

class MerkleTree {
    constructor(leaves) {
        this.leaves = leaves.map(leaf => this.hash(leaf));
        this.tree = this.buildTree(this.leaves);
    }

    hash(data) {
        return crypto.createHash('sha256').update(JSON.stringify(data)).digest('hex');
    }

    buildTree(nodes) {
        if (nodes.length === 1) return nodes;

        const tree = [];
        for (let i = 0; i < nodes.length; i += 2) {
            const left = nodes[i];
            const right = i + 1 < nodes.length ? nodes[i + 1] : left;
            const parent = this.hash(left + right);
            tree.push(parent);
        }

        return this.buildTree(tree);
    }

    getRoot() {
        return this.tree[0];
    }

    verify(leaf, proof) {
        let hash = this.hash(leaf);
        for (const { sibling, position } of proof) {
            hash = position === 'left'
                ? this.hash(sibling + hash)
                : this.hash(hash + sibling);
        }
        return hash === this.getRoot();
    }
}

class ProvenanceChain {
    constructor() {
        this.chain = [this.createGenesisBlock()];
        this.difficulty = 2;
        this.pendingTransactions = [];
        this.chainPath = path.join(__dirname, '../../data/blockchain.json');
        this.loadChain();
    }

    createGenesisBlock() {
        return new Block(0, Date.now(), { type: 'genesis', message: 'SpectraShield Genesis Block' }, '0');
    }

    async loadChain() {
        try {
            const data = await fs.readFile(this.chainPath, 'utf8');
            const savedChain = JSON.parse(data);
            if (savedChain.length > 1) {
                this.chain = savedChain.map(blockData => {
                    const block = new Block(
                        blockData.index,
                        blockData.timestamp,
                        blockData.data,
                        blockData.previousHash
                    );
                    block.hash = blockData.hash;
                    block.nonce = blockData.nonce;
                    return block;
                });
            }
        } catch (error) {
            // Chain file doesn't exist yet, use genesis block
            await this.saveChain();
        }
    }

    async saveChain() {
        try {
            const dir = path.dirname(this.chainPath);
            await fs.mkdir(dir, { recursive: true });
            await fs.writeFile(this.chainPath, JSON.stringify(this.chain, null, 2));
        } catch (error) {
            console.error('Failed to save blockchain:', error);
        }
    }

    getLatestBlock() {
        return this.chain[this.chain.length - 1];
    }

    async addBlock(data) {
        const block = new Block(
            this.chain.length,
            Date.now(),
            data,
            this.getLatestBlock().hash
        );

        block.mineBlock(this.difficulty);
        this.chain.push(block);
        await this.saveChain();

        return block;
    }

    isChainValid() {
        for (let i = 1; i < this.chain.length; i++) {
            const currentBlock = this.chain[i];
            const previousBlock = this.chain[i - 1];

            if (currentBlock.hash !== currentBlock.calculateHash()) {
                return false;
            }

            if (currentBlock.previousHash !== previousBlock.hash) {
                return false;
            }
        }
        return true;
    }

    findBlockByHash(hash) {
        return this.chain.find(block =>
            block.data.videoHash === hash || block.hash === hash
        );
    }

    async recordAnalysis(analysisData) {
        const videoHash = this.hashVideo(analysisData.filePath);

        const blockData = {
            type: 'analysis',
            analysisId: analysisData.analysisId,
            videoHash: videoHash,
            filename: analysisData.filename,
            timestamp: Date.now(),
            uploaderId: analysisData.uploaderId || 'anonymous',
            result: {
                isManipulated: analysisData.result?.isManipulated,
                confidence: analysisData.result?.fakeProbability
            }
        };

        const block = await this.addBlock(blockData);
        return {
            blockHash: block.hash,
            videoHash: videoHash,
            blockIndex: block.index
        };
    }

    hashVideo(filePath) {
        // In production, this would hash the actual video file
        // For now, we'll create a deterministic hash based on filename and timestamp
        return crypto.createHash('sha256')
            .update(filePath + Date.now())
            .digest('hex');
    }

    async verifyVideo(videoHash) {
        const block = this.findBlockByHash(videoHash);

        if (!block) {
            return {
                verified: false,
                message: 'Video hash not found in blockchain'
            };
        }

        const isValid = this.isChainValid();

        return {
            verified: isValid,
            originalUploadHash: block.data.videoHash,
            blockHash: block.hash,
            timestamp: new Date(block.timestamp).toISOString(),
            uploaderId: block.data.uploaderId,
            analysisResult: block.data.result,
            blockIndex: block.index
        };
    }

    async compareVideos(originalHash, newVideoPath) {
        const originalBlock = this.findBlockByHash(originalHash);

        if (!originalBlock) {
            return {
                verified: false,
                message: 'Original video not found'
            };
        }

        const newHash = this.hashVideo(newVideoPath);
        const mismatchScore = this.calculateMismatch(originalBlock.data.videoHash, newHash);

        return {
            verified: originalBlock.data.videoHash === newHash,
            originalUploadHash: originalBlock.data.videoHash,
            newHash: newHash,
            reuploadMismatchScore: mismatchScore,
            timestamp: new Date(originalBlock.timestamp).toISOString()
        };
    }

    calculateMismatch(hash1, hash2) {
        if (hash1 === hash2) return 0;

        // Calculate Hamming distance between hashes
        let mismatch = 0;
        for (let i = 0; i < Math.min(hash1.length, hash2.length); i++) {
            if (hash1[i] !== hash2[i]) mismatch++;
        }

        return (mismatch / hash1.length) * 100;
    }

    getChainStats() {
        return {
            length: this.chain.length,
            isValid: this.isChainValid(),
            latestBlock: this.getLatestBlock().hash,
            totalAnalyses: this.chain.filter(b => b.data.type === 'analysis').length
        };
    }
}

// Singleton instance
let chainInstance = null;

const getBlockchain = () => {
    if (!chainInstance) {
        chainInstance = new ProvenanceChain();
    }
    return chainInstance;
};

module.exports = {
    getBlockchain,
    MerkleTree,
    Block
};
