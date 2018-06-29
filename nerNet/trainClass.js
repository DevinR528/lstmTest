const fs = require('fs');
const path = require('path');
const util = require('util');

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

/**
 *
 *
 * @class TextLSTMNet
 */
class TextLSTMNet {
    /**
     *Creates an instance of TextLSTMNet.
     * @param {String} file location of file to read in
     * @param {String} dataStore location to save model data
     * @param {Number} maxLen length of "sentence" number of characters
     * @param {Number} epochs number of epochs model goes through more is longer more than 10 takes a while with 128 batch
     * @param {Number} lrnRate learning rate of model ex .01
     * @param {Number} batchSize bigger is faster ex 128 splits 150,000 to 1000 batches
     * @param {String} text reading file async
     * @param {Object} charsIndex of shape {character: index}
     * @param {Object} indexChar of shape {index: char}
     * @param {Number} vocab unique character in text
     * @param {Array} sentences sequences of text maxLen long
     * @param {Array} nextChar last char in sequence
     * @param {Tensor} x tensor of shape [[sentencesIndex],[charIndex],[oneHot]]
     * @param {Tensor} y tensor of shape [[sentencesIndex],[nextCharsOneHot]]
     * @memberof TextLSTMNet
     */
    async constructor(file, dataStore, maxLen, epochs, lrnRate, batchSize) {
        this.file = file;
        this.dataStore = 'file://' + dataStore;
        this.maxLen = maxLen;
        this.epochs = epochs;
        this.lrnRate = lrnRate;
        this.batchSize = batchSize;

        this.text = await this.readData();

        [charsIndex, indexChars, vocab] = this.genDict();
        this.charsIndex = charsIndex;
        this.indexChars = indexChars;
        this.vocab = vocab;

        [sentences, nextChars] = this.textToSequence();
        this.sentences = sentences;
        this.nextChars = nextChars;

        [x, y] = this.genTensors();
        this.x = x;
        this.y = y;

    }

    /**
     *
     * @method readData
     * @returns {promise} string of read in file
     * @memberof TrainModel
     */
    readData() {
        return new Promise((resolve, reject) => {
            const readFile = util.promisify(fs.readFile);
            const label = `readData`;
            console.time(label);
            readFile(this.file, 'utf8').then(data => {
                // const cleaned = data.replace(/(\\EOT)/g, '');
                resolve(data);
            });
        })
    }

    /**
     *
     * @method genDict
     * @returns {Array} contains [charToIndex, indexToChar, vocab]
     * @memberof TrainModel
     */
    genDict() {
        const array = this.text.split(/(?!$)/u)
        let i = 0;
        const numKey = {};
        const dict = array.reduce((obj, char) => {
            if (!obj[char]) {
                obj[char] = i;
                numKey[i] = char;
                i++;
            }
            return obj;
        }, {});

        const indexes = Object.keys(dict).length;
        return [dict, numKey, indexes];
    }

    /**
     *
     * @method textToSequence
     * @param {String} text input text (read from file?)
     * @returns {Array} of [sentences] and [last char from seq]
     */
    textToSequence() {
        const strArr = this.text.split(/(?!$)/u);
        const sentences = [];
        const nextChars = [];
        let tempSent, tempNext;
        for (let i = 0; i < strArr.length - this.maxLen; i += 3) {
            tempSent = strArr.slice(i, i + this.maxLen);
            sentences.push(tempSent);

            tempNext = strArr[i + this.maxLen];
            nextChars.push(tempNext);
        }
        console.log(`sentences: ${sentences.length} label: ${nextChars.length}`);
        return [sentences, nextChars];
    }

    /**
     *
     * @method genTensors
     * @param {Array} sentences array of sentence arrays [[characters], []]
     * @param {Number} vocab number of unique characters
     * @param {Array} nextChars last character at maxLength of each sentence in sentences array
     * @param {Object} dict character to index mapping
     * @returns {Array} of Tensors x and y
     * @memberof TrainModel
     */
    genTensors() {

        const xBuffer = tf.buffer([this.sentences.length, this.maxLen, this.vocab]);
        const yBuffer = tf.buffer([this.sentences.length, this.vocab]);
        let sentence, char;
        for (let i = 0; i < this.sentences.length; i++) {
            sentence = this.sentences[i];
            for (let j = 0; j < sentence.length; j++) {
                char = sentence[j]
                xBuffer.set(1, i, j, this.charsIndex[char]);
            }
            yBuffer.set(1, i, dict[this.nextChars[i]]);
        }
        const x = xBuffer.toTensor();
        const y = yBuffer.toTensor();

        return [x, y]
    }

    /**
     *
     * @method train
     * @returns {Promise} Sequential tensorflow model
     * @memberof TrainModel
     */
    async train() {
        const model = tf.sequential();
        model.add(tf.layers.lstm({
            units: 128,
            inputShape: [this.maxLen, this.vocab]
        }));
        model.add(tf.layers.dense({
            units: this.vocab,
            activation: 'softmax'
        }));

        const opt = tf.train.rmsprop(this.lrnRate);
        model.compile({
            loss: 'categoricalCrossentropy',
            optimizer: opt,
            metrics: ['acc']
        });

        console.log("HERE WE GO");
        await model.fit(this.x, this.y, {
            batchSize: this.batchSize,
            epochs: this.epochs,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    // each epoch log acc and loss
                    console.log(`==========================================================================`);
                    console.log(`epoch: ${epoch + 1} | loss: ${logs.loss} | acc: ${logs.acc}`);
                    console.log(`==========================================================================`);
                },
                onBatchEnd: (batch, logs) => {
                    console.log(`Batch: ${batch + 1} | loss: ${logs.loss} | acc: ${logs.acc}`);
                }
            }
        });

        await model.save(this.dataStore);

        // Memory clean up: Dispose the training data.
        xTrain.dispose();
        yTrain.dispose();

        return model
    }

    async buildModel() {
        const model;
        try {
            model = await tf.loadModel(this.dataStore)
        } catch (err) {
            model = await this.train();
        }
        return model;
    }

    _sample(preds, temp = 1) {
        const x = tf.log(preds);
        x.div(temp);
        const expPreds = tf.exp(x);
        const y = tf.sum(expPreds);
        const divPreds = expPreds / y;
        const probas = tf.initializers.randomNormal()
        return tf.argMax(probas);
    }

    async generateText() {
        const model = await tf.loadModel(this.dataStore);

        const sentArr = this.text.split(/\\EOT/g);
        const index = Math.floor(Math.random() * sentArr.length);
        const sentence = sentArr[index];
        console.log(sentence);
        const splitSent = sentence.split(/(?!$)/u);

        let pred;
        for (let t = 0; t < 110; t++) {
            const xPBuff = tf.buffer([1, this.maxLen, this.vocab]);
            for (let i = 0; i < this.maxLen; i++) {
                char = splitSent[i];
                xPBuff.set(1, 0, i, this.charsIndex[char])
            }
            pred = model.predict(xPBuff);
        }
    }
}

const file = path.resolve(__dirname, '../trumpys_tweets.txt');
const dataStore = path.resolve(__dirname, './model-trump-1');
const maxLen = 40; //characters
const epochs = 2;
const lrnRate = 0.01;
const batch = 128

const tm = new TextLSTMNet(file, dataStore, maxLen, epochs, lrnRate, batch);
tm.go();