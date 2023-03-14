const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const fastCsv = require('fast-csv');

const gesture = ['Swipe_Left', 'Swipe_Right', 'Push', 'Clockwise-Circle', 'Anti-Clockwise-Circle'];
const prob_bound = 0.3;

// Options for read CSV directly.
function readCSV(path) {

    const options = {
        headers: false,
    };

    return new Promise((resolve, reject) => {
        var array = [];
        fs.createReadStream(path)
        .pipe(fastCsv.parse(options))
        .on("error", error => reject(error))
        .on('data', row => {
            row = row
                .map(Number)
                .map(a => a.toFixed(4))
                .map(Number)
            array.push(row); 
        })
        .on('end', () => {
            resolve(array);
        });
    });
}


async function doPrediction(){
    var data = await readCSV('fig.csv');
    var inputData = tf.tensor(data)
    var reshapedData = tf.reshape(inputData, [1,100,30]);

    // Load the local model
    const handler = tf.io.fileSystem('path-to-model.json');
    const model = await tf.loadLayersModel(handler);
    var result = model.predict(reshapedData);

    var Argmax = tf.argMax(tf.tensor1d(result.dataSync()).dataSync()); 

    if (result.dataSync()[0] >= prob_bound || result.dataSync()[1] >= prob_bound || result.dataSync()[2] >= prob_bound || result.dataSync()[3] >= prob_bound ||result.dataSync()[4] >= prob_bound){
        console.log(gesture[Argmax.dataSync()]);
    }else{
        console.log("try again"); 
    }
}

doPrediction()
