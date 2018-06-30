const fs = require('fs');
const path = require('path');
const util = require('util');

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node-gpu');

/**
 *
 * @param {String} file location of file to read in
 * @returns {promise} string of read in file
 */
function readData(file) {
    return new Promise((resolve, reject) => {
        const readFile = util.promisify(fs.readFile);
        const label = `readData`;
        console.time(label);
        readFile(file, 'utf8').then(data => {
            // const cleaned = data.replace(/(\\EOT)/g, '');
            resolve(data);
        });
    })
}

/**
 *
 * @param {String} text input text to train on
 * @returns {Array} contains [charToIndex, indexToChar, vocab]
 */
function genDict(text) {
    const array = text.split(/(?!$)/u)
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
 * @param {String} text input text (read from file?)
 * @param {Number} maxLength length of semi random sequence
 * @returns {Array} of [sentences] and [last char from seq]
 */
function textToSequence(text, maxLength) {
    const strArr = text.split(/(?!$)/u);
    const sentences = [];
    const nextChars = [];
    let tempSent, tempNext;
    for (let i = 0; i < strArr.length - maxLength; i += 3) {
        tempSent = strArr.slice(i, i + maxLength);
        sentences.push(tempSent);

        tempNext = strArr[i + maxLength];
        nextChars.push(tempNext);
    }
    console.log(`sentences: ${sentences.length} label: ${nextChars.length}`);
    return [sentences, nextChars];
}

/**
 *
 * @param {Array} sentences array of sentence arrays [[characters], []]
 * @param {Number} maxLength length of "sentence" number of characters
 * @param {Number} vocab number of unique characters
 * @param {Array} nextChars last character at maxLength of each sentence in sentences array
 * @param {Object} dict character to index mapping
 * @returns {Array} of Tensors x and y
 */
function genTensors(sentences, maxLength, vocab, nextChars, dict) {

    const xBuffer = tf.buffer([sentences.length, maxLength, vocab]);
    const yBuffer = tf.buffer([sentences.length, vocab]);
    let sentence, char;
    for (let i = 0; i < sentences.length; i++) {
        sentence = sentences[i];
        for (let j = 0; j < sentence.length; j++) {
            char = sentence[j]
            xBuffer.set(1, i, j, dict[char]);
        }
        yBuffer.set(1, i, dict[nextChars[i]]);
    }
    const x = xBuffer.toTensor();
    const y = yBuffer.toTensor();

    return [x, y]
}

async function train(xTrain, yTrain, maxLen, vocab, batch, epochs, lrnRate, dataFile) {
    const model = tf.sequential();
    model.add(tf.layers.lstm({ units: 128, inputShape: [maxLen, vocab] }));
    model.add(tf.layers.dense({ units: vocab, activation: 'softmax' }));

    const opt = tf.train.rmsprop(lrnRate);
    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: opt,
        metrics: ['acc']
    });

    console.log("HERE WE GO");
    await model.fit(xTrain, yTrain, {
        batchSize: batch,
        epochs: epochs,
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

    await model.save('file://' + dataFile);

    // Memory clean up: Dispose the training data.
    xTrain.dispose()
    yTrain.dispose();
}

const file = path.resolve(__dirname, '../trumpys_tweets.txt');
const dataFile = path.resolve(__dirname, './model-trump-1');
const maxLen = 40; //characters
const epochs = 10;
const lrnRate = 0.1;
const batch = 120
readData(file).then(text => {
    const [charsIndex, indexChars, vocab] = genDict(text);
    const [sentences, nextChars] = textToSequence(text, maxLen);
    const [x, y] = genTensors(sentences, maxLen, vocab, nextChars, charsIndex);
    train(x, y, maxLen, vocab, batch, epochs, lrnRate, dataFile);
});

module.exports = { genDict, readData, maxLen, file, dataFile };