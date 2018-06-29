const fs = require('fs');
const path = require('path');
const util = require('util');

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const TrainModel = require('./trainClass.js');

class NNPrediction {
    constructor() {

    }

    /**
     *
     * @param {String} fileLocation
     * @returns {tensorModel} returns the saved model from trainModel.js
     */
    async getModel(fileLocation) {
        const model = await tf.loadModel('file://' + fileLocation);
        return model
    }
}