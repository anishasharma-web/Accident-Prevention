let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var safeSamples=0, warningSamples=0, dangerSamples=0;
let isPredicting = false;

let capturingInterval = null;
let isCapturingFrame = false;

const btnStartPredict = document.getElementById('startPredicting')
const btnStopPredict = document.getElementById('stopPredicting')

// Load the sound
const warningSound = new Audio('./warning.mp3')
const dangerSound = new Audio('./danger.mp3')

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(3);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 3, activation: 'softmax'})
    ]
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}

function handleCaptureFrame(elem) {
  if (!isCapturingFrame) {
    isCapturingFrame = true;
    capturingInterval = window.setInterval(() => handleButton(elem), 100);
  } else if (isCapturingFrame) {
    window.clearInterval(capturingInterval)
    isCapturingFrame = false;
  }
}

function stopCaptureFrame() {
  
}

function handleButton(elem){
	switch(elem.id){
		case "0":
			safeSamples++;
			document.getElementById("safe-samples").innerText = "Safe Samples: " + safeSamples;
			break;
		case "1":
			warningSamples++;
			document.getElementById("warning-amples").innerText = "Warning samples: " + warningSamples;
			break;
		case "2":
			dangerSamples++;
			document.getElementById("danger-samples").innerText = "Danger samples: " + dangerSamples;
			break;
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);
}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    if (classId == 1) {
      warningSound.play()
      dangerSound.pause()
    } else if (classId === 2) {
      dangerSound.play()
      warningSound.pause()
    }
    else {
      warningSound.pause()
      dangerSound.pause()
    }
    var predictionText = "";
    var predictionClass = "";
    switch(classId){
      case 0:
        predictionText = "The driver is Awake. All OK";
        predictionClass = "prediction-safe";
        break;
      case 1:
        predictionText = "The driver is Potentially sleepy. Take a break";
        predictionClass = "prediction-warning";
        break;
      case 2:
        predictionText = "The driver is Probably asleep. Danger! ";
        predictionClass = "prediction-danger"
        break;
	  }
    const predictionDiv = document.getElementById("prediction");
    predictionDiv.innerText = predictionText;
    predictionDiv.className = `prediction ${predictionClass}`;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
  btnStartPredict.disabled = true;
  btnStopPredict.disabled = true;
	train();
  btnStartPredict.disabled = false;
}

function startPredicting(){
	isPredicting = true;
	predict();
  btnStartPredict.disabled = true;
  btnStopPredict.disabled = false;
}

function stopPredicting(){
	isPredicting = false;
  warningSound.pause()
  dangerSound.pause()
  btnStartPredict.disabled = false;
  btnStopPredict.disabled = true;
  predict();
}

async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
}

init();
