package com.tensorspeech.tensorflowtts.module;

import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

/**
 * @author {@link "mailto:xuefeng.ding@outlook.com" "Xuefeng Ding"}
 * Created 2020-07-20 17:26
 *
 */
public class FastSpeech2 {
    private static final String TAG = "FastSpeech2";
    private OrtSession mModule;
    private OrtEnvironment ortEnv = OrtEnvironment.getEnvironment();

    public FastSpeech2(String modulePath) {
        try {
            // Load model and set number of threads
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.setIntraOpNumThreads(5);
            mModule = ortEnv.createSession(modulePath);
            Map<String, NodeInfo> inputInfo = mModule.getInputInfo();
            for (Map.Entry<String, NodeInfo> entry : inputInfo.entrySet()) {
                String inputName = entry.getKey();
                TensorInfo tensorInfo = (TensorInfo)entry.getValue().getInfo();
                Log.d(TAG, "input:"
                        + " name:" + inputName
                        + " shape:" + Arrays.toString(tensorInfo.getShape()) +
                        " dtype:" + tensorInfo.onnxType.toString());
            }
            Map<String, NodeInfo> outputInfo = mModule.getOutputInfo();
            for (Map.Entry<String, NodeInfo> entry : outputInfo.entrySet()) {
                String outputName = entry.getKey();
                TensorInfo tensorInfo = (TensorInfo)entry.getValue().getInfo();
                Log.d(TAG, "output:"
                        + " name:" + outputName
                        + " shape:" + Arrays.toString(tensorInfo.getShape()) +
                        " dtype:" + tensorInfo.onnxType.toString());
            }
            Log.d(TAG, "successfully init");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public FloatBuffer getMelSpectrogram(int[] inputIds, float speed) {
        try {
            Log.d(TAG, "input id length: " + inputIds.length);
            Map<String, OnnxTensor> feed = new HashMap<>();

            ByteBuffer input_ids_byteBuffer = ByteBuffer.allocate(1 * inputIds.length * 4);
            IntBuffer input_ids_intBuffer = input_ids_byteBuffer.asIntBuffer();
            input_ids_intBuffer.put(inputIds);
            OnnxTensor input_ids_tensor = createInputTensor(TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, new long[] {1, inputIds.length}, input_ids_byteBuffer);
            feed.put("input_ids", input_ids_tensor);
            Log.d(TAG, "input1 end: ");

            int[] attention_mask = new int[]{0};
            ByteBuffer attention_mask_byteBuffer = ByteBuffer.allocate(1 * 4);
            IntBuffer attention_mask_intBuffer = attention_mask_byteBuffer.asIntBuffer();
            attention_mask_intBuffer.put(attention_mask);
            OnnxTensor attention_mask_tensor = createInputTensor(TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, new long[] {1, 1}, attention_mask_byteBuffer);
            feed.put("attention_mask", attention_mask_tensor);
            Log.d(TAG, "input2 end: ");

            int[] speaker_ids = new int[]{0};
            ByteBuffer speaker_ids_byteBuffer = ByteBuffer.allocate(1 * 4);
            IntBuffer speaker_ids_intBuffer = speaker_ids_byteBuffer.asIntBuffer();
            speaker_ids_intBuffer.put(speaker_ids);
            OnnxTensor speaker_ids_tensor = createInputTensor(TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, new long[] {1}, speaker_ids_byteBuffer);
            feed.put("speaker_ids", speaker_ids_tensor);
            Log.d(TAG, "input3 end: ");

            float[] speed_ratios = new float[]{speed};
            ByteBuffer speed_ratios_byteBuffer = ByteBuffer.allocate(1 * 4);
            FloatBuffer speed_ratios_floatBuffer = speed_ratios_byteBuffer.asFloatBuffer();
            speed_ratios_floatBuffer.put(speed_ratios);
            OnnxTensor speed_ratios_tensor = createInputTensor(TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, new long[] {1}, speed_ratios_byteBuffer);
            feed.put("speed_ratios", speed_ratios_tensor);
            Log.d(TAG, "input4 end: ");

            float[] f0_ratios = new float[]{1F};
            ByteBuffer f0_ratios_byteBuffer = ByteBuffer.allocate(1 * 4);
            FloatBuffer f0_ratios_floatBuffer = f0_ratios_byteBuffer.asFloatBuffer();
            f0_ratios_floatBuffer.put(f0_ratios);
            OnnxTensor f0_ratios_tensor = createInputTensor(TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, new long[] {1}, f0_ratios_byteBuffer);
            feed.put("f0_ratios", f0_ratios_tensor);
            Log.d(TAG, "input5 end: ");

            float[] energy_ratios = new float[]{1F};
            ByteBuffer energy_ratios_byteBuffer = ByteBuffer.allocate(1 * 4);
            FloatBuffer energy_ratios_floatBuffer = energy_ratios_byteBuffer.asFloatBuffer();
            energy_ratios_floatBuffer.put(energy_ratios);
            OnnxTensor energy_ratios_tensor = createInputTensor(TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, new long[] {1}, energy_ratios_byteBuffer);
            feed.put("energy_ratios", energy_ratios_tensor);
            Log.d(TAG, "input6 end: ");

            long inputs = mModule.getNumInputs();
            Log.d(TAG, "NumInputs " + inputs);

            OrtSession.Result result = mModule.run(feed, new OrtSession.RunOptions());
            Log.d(TAG, "Result end");
            Optional<OnnxValue> onnxValue = result.get("Identity");
            if (onnxValue.isPresent()) {
                OnnxTensor onnxTensor = (OnnxTensor)onnxValue.get();
                FloatBuffer outputBuffer = onnxTensor.getFloatBuffer();
                Log.d(TAG, "outputBuffer position: " + outputBuffer.remaining());
                Log.d(TAG, "OnnxValue isPresent: ");
                return outputBuffer;
            } else {
                Log.d(TAG, "OnnxValue is not Present: ");
                return null;
            }
        } catch (Exception e) {
            e.printStackTrace();
            Log.d(TAG, "OnnxValue Exception ");
            return null;
        }
    }

    private OnnxTensor createInputTensor(TensorInfo.OnnxTensorType tensorType, long[] dims, ByteBuffer values) throws Exception {
        OnnxTensor tensor = null;
        switch (tensorType) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                FloatBuffer buffer = values.asFloatBuffer();
                tensor = OnnxTensor.createTensor(ortEnv, buffer, dims);
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                IntBuffer buffer = values.asIntBuffer();
                tensor = OnnxTensor.createTensor(ortEnv, buffer, dims);
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                LongBuffer buffer = values.asLongBuffer();
                tensor = OnnxTensor.createTensor(ortEnv, buffer, dims);
                break;
            }
            default:
                throw new IllegalStateException("Unexpected value: " + tensorType.toString());
        }

        return tensor;
    }
}
