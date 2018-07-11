const fs = require('fs');
const path = require('path');
const util = require('util');

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

/**
 *
 *
 * @class TextLSTM
 */
class TextLSTM {
    /**
     *Creates an instance of TextLSTM must call init() also
     * @param {String} file location of file to read in
     * @param {String} dataStore location to save model data
     * @param {Number} maxLen length of "sentence" number of characters
     * @param {Number} epochs number of epochs model goes through more is longer more than 10 takes a while with 128 batch
     * @param {Number} lrnRate learning rate of model ex .01
     * @param {Number} batchSize bigger is faster ex 128 splits 150,000 to 1000 batches
     * @memberof TextLSTMNet
     */
    constructor(file, dataStore, maxLen, epochs, lrnRate, batchSize) {
        this.file = path.resolve(__dirname, file);
        this.dataStore = 'file://' + path.resolve(__dirname, dataStore);
        this.maxLen = maxLen;
        this.epochs = epochs;
        this.lrnRate = lrnRate;
        this.batchSize = batchSize;

    }

    /**
     *
     * @method init must call to read in data and create tensors
     * @property {String} this.text reading file async
     * @property {Object} this.charsIndex of shape {character: index}
     * @property {Object} this.indexChar of shape {index: char}
     * @property {Number} this.vocab unique character in text
     * @property {Array} this.sentences sequences of text maxLen long
     * @property {Array} this.nextChar last char in sequence
     * @property {Tensor} this.x tensor of shape [[sentencesIndex],[charIndex],[oneHot]]
     * @property {Tensor} this.y tensor of shape [[sentencesIndex],[nextCharsOneHot]]
     * @memberof TextLSTMNet
     */
    async init(){
        this.text = await this.readData();

        const [charsIndex, indexChars, vocab] = this.genDict();
        this.charsIndex = charsIndex;
        this.indexChars = indexChars;
        this.vocab = vocab;

        const [sentences, nextChars] = this.textToSequence();
        this.sentences = sentences;
        this.nextChars = nextChars;

        const [x, y] = this.genTensors();
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
            yBuffer.set(1, i, this.charsIndex[this.nextChars[i]]);
        }
        const x = xBuffer.toTensor();
        const y = yBuffer.toTensor();

        return [x, y]
    }

    /**
     *
     * @method _train
     * @returns {Promise} Sequential tensorflow model
     * @memberof TrainModel
     */
    async _train() {
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

        return model
    }

    async trainOrLoadModel() {
        try {           
            const model = await tf.loadModel(this.dataStore + '/model.json');
            console.log('loading');
            return model;
        } catch (err) {
            console.log('training');
            const model = await this._train();
            return model;
        }
        
    }

    _sample(predictions=tf.tensor2d, temperature=1) {
        const temp = tf.scalar(temperature);
        
        const prob = tf.max(predictions);
        const index = prob.dataSync();
        
        const arr = predictions.flatten();
        const probArr = Array.from(arr.dataSync());
        //tf.argMax(arr).print();
        let numIndex;
        probArr.find((val, i) => {
            if(val === index[0]){
                numIndex = i;
            }
        })

        //console.log(numIndex);
        return numIndex;
        
        // let x = tf.log(predictions);
        // x = x.div(temp);
        // const expPreds = tf.exp(x);
        // const y = tf.sum(expPreds);
        // x = expPreds.div(y)
        // x.print(true);
        // const probs = tf.multinomial(x, );
        // tf.argMax(probs).print();
    }

    /**
     *
     *
     * @param {*} lenTextGen
     * @param {*} model
     * @memberof TextLSTMNet
     */
    // TODO wrap predict in tidy()
    generateText(lenTextGen, model) {
        const sentArr = this.text.split(/\\EOT/g);
        const index = Math.floor(Math.random() * sentArr.length);
        const sentence = sentArr[index];
        const splitSent = sentence.split(/(?!$)/u);

        console.log(splitSent[this.maxLen]);
        console.log(sentence);

        let charArr = [];
        let predictArr = [];
        charArr = splitSent.slice(0, this.maxLen);
        predictArr = splitSent.slice(0, this.maxLen);
    
        let pred, char;
        for (let t = 0; t < lenTextGen; t++) {
            tf.tidy(() => {
                const xPBuff = tf.buffer([1, charArr.length, this.vocab]);
                for (let i = 0; i < charArr.length; i++) {
                    char = charArr[i];
                    xPBuff.set(1, 0, i, this.charsIndex[char])
                }
                const x = xPBuff.toTensor();
                pred = model.predict(x);
                const nextIndex = this._sample(pred);

                charArr.shift();
                charArr.push(this.indexChars[nextIndex]);
                predictArr.push(this.indexChars[nextIndex]);
                
            })
        }
        const pridString = predictArr.join('');
        console.log(pridString);
        
    }

    async reTrainModel(locationToSave=this.dataStore){
        const model = await tf.loadModel(this.dataStore + '/model.json')

        const opt = tf.train.rmsprop(this.lrnRate);
        model.compile({
            loss: 'categoricalCrossentropy',
            optimizer: opt,
            metrics: ['acc']
        });

        await model.fit(this.x, this.y, {
            batchSize: this.batchSize,
            epochs: this.epochs,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    // each epoch log acc and loss
                    const msg = (`==========================================================================
                    epoch: ${epoch + 1} | loss: ${logs.loss} | acc: ${logs.acc}
                    ==========================================================================`);

                    console.log(msg);
                },
                onBatchEnd: (batch, logs) => {
                    // batch log acc and loss
                        console.log(`Batch: ${batch + 1} | loss: ${logs.loss} | acc: ${logs.acc}`);
                }
            }
        });
        await model.save('file://' + locationToSave);
    }
}


(async function run() {

    const file = '../trumpys_tweets.txt';
    const dataStore = './model-trump-2';
    const maxLen = 40; //characters
    const epochs = 2;
    const lrnRate = 0.01;
    const batch = 128;

    const tm = new TextLSTM(file, dataStore, maxLen, epochs, lrnRate, batch);
    await tm.init();
    // const model = await tm.trainOrLoadModel();
    // tm.generateText(110, model);
    for (let i = 0; i < 2; i++) {
        await tm.reTrainModel(path.resolve(__dirname, 'model-trump-2'));
    }
})();
