package com.tensorspeech.tensorflowtts.module;

import android.util.Log;

import java.nio.FloatBuffer;
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
public class MBMelGan {
    private static final String TAG = "MBMelGan";
    private OrtSession mModule;

    private OrtEnvironment ortEnv = OrtEnvironment.getEnvironment();

    public MBMelGan(String modulePath) {
        try {
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.setIntraOpNumThreads(5);
            sessionOptions.setInterOpNumThreads(5);
            mModule = ortEnv.createSession(modulePath, sessionOptions);
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

    public float[] getAudio(FloatBuffer input) {
        try {
            Log.d(TAG, "getAudio start ");
            Map<String, OnnxTensor> feed = new HashMap<>();
            Log.d(TAG, "Input Position " + input.remaining());
            long[] dims = new long[] {1, input.remaining()/80, 80};
            OnnxTensor tensor = OnnxTensor.createTensor(ortEnv, input, dims);
            Log.d(TAG, "tensor created ");
            feed.put("mels", tensor);
            OrtSession.Result result = mModule.run(feed, new OrtSession.RunOptions());
            Log.d(TAG, "inference done ");
            Optional<OnnxValue> onnxValue = result.get("Identity");
            if (onnxValue.isPresent()) {
                OnnxTensor onnxTensor = (OnnxTensor)onnxValue.get();
                FloatBuffer outputBuffer = onnxTensor.getFloatBuffer();
                float[] audioArray = new float[outputBuffer.remaining()];
                outputBuffer.get(audioArray);
                Log.d(TAG, "OnnxValue Is Present ");
                return audioArray;
            } else {
                Log.d(TAG, "OnnxValue Is not Present ");
                return null;
            }
        } catch (Exception e) {
            e.printStackTrace();
            Log.d(TAG, "OnnxValue Exception");
            return null;
        }
    }
}
