package XOR;

import java.util.Random;

public class NeuralNetwork {
	private double[] inputLayer;
	private double[][] inputWeights;
	private double[][] inputWeightsAdjust;
	private double[] hiddenLayer;
	private double[][] hiddenWeights;
	private double[][] hiddenWeightsAdjust;
	private double[] outputLayer;
	
	private double[][] inputs;
	private double[][] outputs;
	
	private int testNum = 0;
	
	private double learningRate = 0.25;
	
	
	public NeuralNetwork(double[][] inputs, double[][] outputs, int inputLayerSize, int hiddenLayerSize, int outputLayerSize) {
		this.inputs = inputs;
		this.outputs = outputs;
		initLayers(inputLayerSize,hiddenLayerSize,outputLayerSize);
		initWeightsArr(inputLayerSize,hiddenLayerSize,outputLayerSize);
		initWeightsVal(-1,1);
		setInputLayer(inputs[testNum]);
	}
	
	public void testEveryLayer() {
		for(int i=0;i<inputs.length;i++) {
			System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
			setInputLayer(inputs[i]);
			feedForward();
			display(i);
		}
	}
	
	public void learn(int iter) {
		System.out.println("Iteration: " + (1));
		display(testNum);
		for(int i=0;i<iter;i++) {
			
			if((i+1)%200 == 0) {
				System.out.println("---------------------------------");
				System.out.println("Iteration: " + (i+1));
				display(testNum);
			}
			feedForward();
			backProp(outputs[testNum][0]);
		}
	}
	
	private void display(int i) {
		
		System.out.println("Input Layer:");
		displayMatrix(inputLayer);
		
		//System.out.println("Hidden Layer:");
		//displayMatrix(hiddenLayer);
		
		System.out.println("Output Layer:");
		displayMatrix(outputLayer);
		
		System.out.println("Desired Output: ");
		displayMatrix(outputs[i]);
	}
	
	private void feedForward() {
		for(int h=0;h<hiddenLayer.length;h++) {
			double value = 0;
			for(int i=0;i<inputLayer.length;i++) {
				value += (inputLayer[i]*inputWeights[i][h]);
			}
			//System.out.println(value);
			hiddenLayer[h] = sigmoid(value);
		}
		for(int o=0;o<outputLayer.length;o++) {
			double value = 0;
			for(int h=0;h<hiddenLayer.length;h++) {
				value += (hiddenLayer[h]*hiddenWeights[h][o]);
			}
			//System.out.println(value);
			outputLayer[o] = sigmoid(value);
		}
		
		//testOutput = outputLayer[0]*1.2;
		//System.out.println("Test desired output: " + testOutput);
		//double error = errorCalculate(testOutput,outputLayer[0]);
		//System.out.println("Error: " + error);
	}
	
	private void backProp(double targetOut) {
		adjustHiddenWeights(targetOut);
		adjustInputWeights(targetOut);
		
	}
	
	private void adjustInputWeights(double targetOut) {
		for(int i=0;i<inputWeightsAdjust.length;i++) {
			for(int j=0;j<inputWeightsAdjust[0].length;j++) {
				double out = outputLayer[0];
				double target = targetOut;
				double h = hiddenLayer[j];
				double in = inputLayer[i];
				inputWeightsAdjust[i][j] = (((out-target) * (out*(1.0-out))*h) * (h*(1.0-h)) * in);
			}
		}
		
		for(int i=0;i<inputWeights.length;i++) {
			for(int j=0;j<inputWeights[0].length;j++) {
				inputWeights[i][j] -= inputWeightsAdjust[i][j] * learningRate;
			}
		}
	}
	
	private void adjustHiddenWeights(double targetOut) {
		for(int i=0;i<hiddenWeightsAdjust.length;i++) {
			double out = outputLayer[0];
			double target = targetOut;
			double h = hiddenLayer[i];
			hiddenWeightsAdjust[i][0] = ((out-target) * (out*(1.0-out))*h);
		}
		
		for(int i=0;i<hiddenWeights.length;i++) {
			for(int j=0;j<hiddenWeights[0].length;j++) {
				hiddenWeights[i][0] -= hiddenWeightsAdjust[i][0] * learningRate;
			}
		}
	}
	
	private void setInputLayer(double[] in) {
		if(in.length == inputLayer.length) {
			inputLayer = in;
		}
		else {
			System.out.println("ERROR ~ NeuralNetwork.java : setInputLayer(double[] in) in.length does not match inputLayer.length");
		}
	}
	
	private void initLayers(int inSize, int hiddenSize, int outputSize) {
		inputLayer = new double[inSize];
		hiddenLayer = new double[hiddenSize];
		outputLayer = new double[outputSize];
	}
	
	private void initWeightsArr(int inSize, int hiddenSize, int outputSize) {
		inputWeights = new double[inSize][hiddenSize];
		inputWeightsAdjust = new double[inSize][hiddenSize];
		hiddenWeights = new double[hiddenSize][outputSize];
		hiddenWeightsAdjust = new double[hiddenSize][outputSize];
	}
	private void initWeightsVal(double lower, double upper) {
		for(int i=0;i<inputWeights.length;i++) {
			for(int j=0;j<inputWeights[0].length;j++) {
				inputWeights[i][j] = randomDouble(lower,upper);
			}
		}
		for(int i=0;i<hiddenWeights.length;i++) {
			for(int j=0;j<hiddenWeights[0].length;j++) {
				hiddenWeights[i][j] = randomDouble(lower,upper);
			}
		}
	}
	
	private double sigmoid(double value) {
		return (1.0 / (1.0 + Math.exp(-value)));
	}
	
	private double sigmoidDerivative(double value) {
		return sigmoid(value)*(1 - sigmoid(value));
	}
	
	private double errorCalculate(double desired, double output) {
		return 0.5*Math.pow((desired-output), 2);
	}
	
	private double errorDerivative(double desired, double output) {
		return output - desired;
	}
	
	private double randomDouble(double min, double max) { 
		Random rand = new Random();
		return (rand.nextDouble() * (max - min) + min);
	}
	
	private void displayMatrix(double[] matrix) {
		for(int i=0;i<matrix.length;i++) {
			System.out.print("{ ");
			System.out.print(matrix[i]);
			System.out.println(" }");
		}
	}
	
	private void displayMatrix(double[][] matrix) {
		for(int i=0;i<matrix.length;i++) {
			System.out.print("{ ");
			for(int j=0;j<matrix[0].length;j++) {
				System.out.print(matrix[i][j] +" ");
			}
			System.out.println("}");
		}
	}
}
