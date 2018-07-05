const {
    Worker,
    isMainThread,
    parentPort,
    workerData
} = require('worker_threads');
const fs = require('fs');
const path = require('path');
const util = require('util');

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

class Worker {
    constructor(batch) {
        this.batch = batch;

    }

    async startThread() {
        if (isMainThread) {
            tf.seq

        } else {

        }
    }
}