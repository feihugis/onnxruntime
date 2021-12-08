# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest
import os
import numpy as np
import gc

import onnxruntime as onnxrt
import threading
import sys
from helper import get_name
from onnxruntime.capi.onnxruntime_pybind11_state import Fail
import time

class TestInferenceSession(unittest.TestCase):
  # def testOrtValue(self):

  #       numpy_arr_input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
  #       numpy_arr_output = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)

  #       def test_session_with_ortvalue_input(ortvalue):
  #           sess = onnxrt.InferenceSession(get_name("mul_1.onnx"),
  #                                          providers=onnxrt.get_available_providers())
  #           res = sess.run(["Y"], {"X": ortvalue})
  #           self.assertTrue(np.array_equal(res[0], numpy_arr_output))

  #       ortvalue1 = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input)
  #       self.assertEqual(ortvalue1.device_name(), "cpu")
  #       self.assertEqual(ortvalue1.shape(), [3, 2])
  #       self.assertEqual(ortvalue1.data_type(), "tensor(float)")
  #       self.assertEqual(ortvalue1.is_tensor(), True)
  #       self.assertTrue(np.array_equal(ortvalue1.numpy(), numpy_arr_input))

  #       # Pass in the constructed OrtValue to a session via Run() and check results
  #       test_session_with_ortvalue_input(ortvalue1)

  #       # The constructed OrtValue should still be valid after being used in a session
  #       self.assertTrue(np.array_equal(ortvalue1.numpy(), numpy_arr_input))

  #       if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
  #           print("CUDAExecutionProvider++++")
  #           ortvalue2 = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input, 'cuda', 0)
  #           self.assertEqual(ortvalue2.device_name(), "cuda")
  #           self.assertEqual(ortvalue2.shape(), [3, 2])
  #           self.assertEqual(ortvalue2.data_type(), "tensor(float)")
  #           self.assertEqual(ortvalue2.is_tensor(), True)
  #           self.assertTrue(np.array_equal(ortvalue2.numpy(), numpy_arr_input))

  #           # Pass in the constructed OrtValue to a session via Run() and check results
  #           test_session_with_ortvalue_input(ortvalue2)

  #           # The constructed OrtValue should still be valid after being used in a session
  #           self.assertTrue(np.array_equal(ortvalue2.numpy(), numpy_arr_input))
            
  # def testRunModelWithCudaCopyStream(self):
  #       available_providers = onnxrt.get_available_providers()

  #       if (not 'CUDAExecutionProvider' in available_providers):
  #           print("Skipping testRunModelWithCudaCopyStream when CUDA is not available")
  #       else:
  #           # adapted from issue #4829 for a race condition when copy is not on default stream
  #           # note:
  #           # 1. if there are intermittent failure in this test, something is wrong
  #           # 2. it's easier to repro on slower GPU (like M60, Geforce 1070)

  #           # to repro #4829, set the CUDA EP do_copy_in_default_stream option to False
  #           providers = [("CUDAExecutionProvider", {"do_copy_in_default_stream": True}), "CPUExecutionProvider"]

  #           session = onnxrt.InferenceSession(get_name("issue4829.onnx"), providers=providers)
  #           shape = np.array([2,2], dtype=np.int64)
  #           for iteration in range(100000):
  #               result = session.run(output_names=['output'], input_feed={'shape': shape})
  def testOrtValueUpdateInPlace(self):
      x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
      ortvalue_cpu = onnxrt.OrtValue.ortvalue_from_numpy(x)
      ortvalue_gpu = onnxrt.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
      np.testing.assert_allclose(x, ortvalue_cpu.numpy())
      np.testing.assert_allclose(x, ortvalue_gpu.numpy())
      
      x = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
      ortvalue_cpu.update_inplace(x)
      ortvalue_gpu.update_inplace(x)
      np.testing.assert_allclose(x, ortvalue_cpu.numpy())
      np.testing.assert_allclose(x, ortvalue_gpu.numpy())
      
  def testRunModelWithCudaGraph(self):
      providers = ["CUDAExecutionProvider"]
      sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=providers)
      x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
      y = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
      x_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
      y_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(y, 'cuda', 0)
      
      feeds = {"X": x_ortvalue}
      fetches = {"Y": y_ortvalue}
      
      for _ in range(20):
        sess.run_with_feeds_fetches_ort_values(feeds, fetches)
      
      y_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
      np.testing.assert_allclose(y_expected, fetches["Y"].numpy(), rtol=1e-05, atol=1e-05)
      
      
      sess.turn_on_capture()      
      sess.run_with_feeds_fetches_ort_values(feeds, fetches)
      sess.turn_off_capture()
      sess.replay()
      np.testing.assert_allclose(y_expected, fetches["Y"].numpy(), rtol=1e-05, atol=1e-05)
      
      x = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
      y_expected = np.array([[10.0, 40.0], [90.0, 160.0], [250.0, 360.0]], dtype=np.float32)
      x_ortvalue.update_inplace(x)
      sess.replay()
      np.testing.assert_allclose(y_expected, fetches["Y"].numpy(), rtol=1e-05, atol=1e-05)
                
      
                    
  def a_testCUDAGraphCapture(self):
      print(f"PID: {os.getpid()} \n")
      
      # onnxrt.set_default_logger_severity(0)
      providers = [("CUDAExecutionProvider", {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',})]
      sess_options = onnxrt.SessionOptions()
      sess_options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
      sess_options.graph_optimization_level = onnxrt.GraphOptimizationLevel(99)
      sess_options.enable_cpu_mem_arena = False
      sess_options.log_verbosity_level = 4
      sess_options.add_session_config_entry("use_device_allocator_for_initializers", "1")
      sess_options.add_session_config_entry("session_log_severity_level", "0")
      
      run_options = onnxrt.RunOptions()
      run_options.log_verbosity_level = 1
      
      
      
      src_tokens = np.array([[0,99,99,99,99,99,99,99,99,99,99,99,99,99,99,2]])
      prev_out_tokens = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,]])
      
      src_tokens_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(src_tokens, 'cuda', 0)
      # src_tokens_ortvalue = onnxrt.OrtValue.ortvalue_from_shape_and_type([1, 16], element_type=np.longlong, device_type='cuda', device_id=0)
      prev_out_tokens_ortvalue  = onnxrt.OrtValue.ortvalue_from_numpy(prev_out_tokens, 'cuda', 0)
      src_lengths_ortvalue  = onnxrt.OrtValue.ortvalue_from_numpy(np.array([16]), 'cuda', 0)
      K = 200
      topk_probs_ortvalue = onnxrt.OrtValue.ortvalue_from_shape_and_type(
        [16, K], element_type=np.float32, device_type='cuda', device_id=0)
      topk_index_ortvalue = onnxrt.OrtValue.ortvalue_from_shape_and_type(
        [16, K], element_type=np.longlong, device_type='cuda', device_id=0)
      
      # sess_options.add_initializer("src_tokens", src_tokens_ortvalue)
      # sess_options.add_initializer("prev_out_tokens", prev_out_tokens_ortvalue)
      # sess_options.add_initializer("src_lengths", src_lengths_ortvalue)
            
      feeds = {
          "src_tokens": src_tokens_ortvalue,
          "src_lengths": src_lengths_ortvalue,
          "prev_out_tokens": prev_out_tokens_ortvalue}
      fetches = {
          "topk_probs": topk_probs_ortvalue,
          "topk_index": topk_index_ortvalue}
            
      
      # model_path = "/model/data/onnx-opt/model_optimized_v3.onnx"
      model_path = "/model/data/onnx-opt/model_onnx_debug_base_fp32.onnx"
      # model_path = "/model/data/onnx-opt/model_onnx_debug_base_fp16_1.onnx"
      session = onnxrt.InferenceSession(model_path, sess_options=sess_options, providers=providers)
      
      print("Finish session loading  \n")
      
      
      res = session.run(
          ["topk_probs", "topk_index"],
           feeds)
      print("Baseline: \n", res)
      
      for _ in range(1):
          session.run_with_feeds_fetches_ort_values(feeds, fetches, run_options)
      print("Capturing infer results: \n", fetches['topk_probs'].numpy())
          
      print("Finish warmup \n")
  
      session._sess.turn_on_capture()      
      session.run_with_feeds_fetches_ort_values(feeds, fetches, run_options)
      session._sess.turn_off_capture()
      session._sess.replay()
      print("Replay infer result: \n", fetches['topk_probs'].numpy())
      
      print("Update scr_tokens \n")
      src_tokens = np.array([[101, 2040, 102, 101, 2003, 102, 101, 3419, 102, 101, 8592, 102, 102, 0, 0, 0]])
      src_tokens_ortvalue._ortvalue.update_inplace(src_tokens)
      print(src_tokens_ortvalue.numpy())
      
      
      REPEAT = 100
      t0 = time.time()
      for _ in range(REPEAT):
        src_tokens = np.array([[101, 2040, 102, 101, 2003, 102, 101, 3419, 102, 101, 8592, 102, 102, 0, 0, 0]])
        src_tokens_ortvalue._ortvalue.update_inplace(src_tokens)
        session._sess.replay();
      t1 = time.time()
      print(fetches['topk_probs'].numpy())
      print(f"Cuda Graph avg time: {(t1 - t0)/REPEAT} \n")


if __name__ == '__main__':
    unittest.main()
