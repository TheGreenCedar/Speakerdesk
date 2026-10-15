"""Executable attention-boundary contracts; no model fixture or acceptance changes."""
import copy
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

from fuse_attention import SCALE, fuse_attention
from compare_mha_probe import check_attention_placement


def fixture():
    """Preprojected inputs keep this check at the changed attention boundary."""
    nodes, initializers, biases = [], [], []
    rng = np.random.default_rng(9124)
    for index in range(24):
        prefix = f"/model/encoder/layers.{index}/self_attn/"
        hidden = 'x' if index == 0 else f"/model/encoder/layers.{index-1}/self_attn/Reshape_9_output_0"

        def node(name, op, inputs, **attributes):
            output = prefix + name + '_output_0'
            nodes.append(helper.make_node(op, inputs, [output], name=prefix+name, **attributes))
            return output

        projected, layer_biases = [], []
        for projection in ('q','k','v'):
            bias = f'model.encoder.layers.{index}.self_attn.{projection}_proj.bias'
            value = rng.normal(0, .15, (1024,)).astype(np.float32)
            layer_biases.append(value)
            initializers.append(numpy_helper.from_array(value, bias))
            before_bias = node(projection+'_proj/MatMul', 'Identity', [hidden])
            projected.append(node(projection+'_proj/Add', 'Add', [bias, before_bias]))
        biases.append(layer_biases)
        initializers.extend([
            numpy_helper.from_array(np.array([0,-1,16,64], np.int64), prefix+'shape4'),
            numpy_helper.from_array(np.array([0,-1,1024], np.int64), prefix+'shape3'),
            numpy_helper.from_array(np.array(np.sqrt(.125),np.float32), prefix+'sdpa/Sqrt_1_output_0'),
            numpy_helper.from_array(np.array(np.sqrt(.125),np.float32), prefix+'sdpa/Sqrt_2_output_0'),
        ])
        reshaped = [node(n,'Reshape',[value,prefix+'shape4'],allowzero=0)
                    for n,value in zip(['Reshape_2','Reshape_5','Reshape_8'],projected)]
        q = node('sdpa/Transpose','Transpose',[reshaped[0]],perm=[0,2,1,3])
        k = node('sdpa/Transpose_2','Transpose',[reshaped[1]],perm=[0,2,3,1])
        v = node('sdpa/Transpose_1','Transpose',[reshaped[2]],perm=[0,2,1,3])
        q = node('sdpa/Mul','Mul',[q,prefix+'sdpa/Sqrt_1_output_0'])
        k = node('sdpa/Mul_1','Mul',[k,prefix+'sdpa/Sqrt_2_output_0'])
        scores = node('sdpa/MatMul','MatMul',[q,k])
        weights = node('sdpa/Softmax','Softmax',[scores],axis=-1)
        context = node('sdpa/MatMul_1','MatMul',[weights,v])
        joined = node('sdpa/Transpose_3','Transpose',[context],perm=[0,2,1,3])
        node('Reshape_9','Reshape',[joined,prefix+'shape3'],allowzero=0)
    graph = helper.make_graph(nodes,'attention_contract',
        [helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,'time',1024])],
        [helper.make_tensor_value_info(nodes[-1].output[0],TensorProto.FLOAT,[1,'time',1024])],
        initializer=initializers)
    model = helper.make_model(graph,opset_imports=[helper.make_opsetid('',14)])
    model.ir_version = 7
    return model, biases


def reference(x, biases):
    for q_bias,k_bias,v_bias in biases:
        def heads(value):
            return value.reshape(1,-1,16,64).transpose(0,2,1,3)
        # Independent source formula: bias is included before query scale.
        q, k, v = heads((x+q_bias)*np.float32(.125)), heads(x+k_bias), heads(x+v_bias)
        scores = q @ k.transpose(0,1,3,2)
        weights = np.exp(scores-scores.max(-1,keepdims=True))
        weights /= weights.sum(-1,keepdims=True)
        x = (weights @ v).transpose(0,2,1,3).reshape(1,-1,1024)
    return x


class AttentionContracts(unittest.TestCase):
    def test_unnamed_exported_score_tail_is_preserved(self):
        model,_=fixture()
        logits=model.graph.output[0].name
        tail=[helper.make_node('LogSoftmax',[logits],['log_probabilities'],axis=-1),
              helper.make_node('Gather',['log_probabilities','token_ids'],['scores'],axis=-1)]
        model.graph.node.extend(tail)
        model.graph.initializer.append(numpy_helper.from_array(np.array([0,5,1023],np.int64),'token_ids'))
        del model.graph.output[:]
        model.graph.output.append(helper.make_tensor_value_info('scores',TensorProto.FLOAT,[1,'time',3]))
        before=[n.SerializeToString() for n in tail]
        fuse_attention(model)
        self.assertEqual([n.SerializeToString() for n in model.graph.node[-2:]],before)
        self.assertEqual(sum(n.op_type=='MultiHeadAttention' for n in model.graph.node),24)
        onnx.checker.check_model(model)

    def test_other_gpu_work_cannot_hide_cpu_attention_fallback(self):
        gpu = {'args': {'provider':'DmlExecutionProvider','op_name':'DmlFusedNode'}}
        for operator in ['Attention','MultiHeadAttention']:
            mixed = [gpu, {'args': {'provider':'CPUExecutionProvider','op_name':operator}}]
            with self.assertRaisesRegex(ValueError,'fell back to CPU'):
                check_attention_placement(mixed,'dml')
            check_attention_placement(mixed,'cpu')
        check_attention_placement([gpu,{'args':{'provider':'CPUExecutionProvider','op_name':'Shape'}}],'dml')

    def test_dynamic_attention_preserves_full_bias_scale_layout_and_parameters(self):
        model, biases = fixture()
        parameter_bytes = [p.SerializeToString() for p in model.graph.initializer]
        unchanged = {n.name:n.SerializeToString() for n in model.graph.node if '_proj/' in n.name}
        fuse_attention(model)
        self.assertEqual([p.SerializeToString() for p in model.graph.initializer[:-1]],parameter_bytes)
        self.assertEqual({n.name:n.SerializeToString() for n in model.graph.node if '_proj/' in n.name},unchanged)
        self.assertEqual(sum(n.op_type=='MultiHeadAttention' for n in model.graph.node),24)
        self.assertFalse(any('/sdpa/' in n.name for n in model.graph.node))
        self.assertEqual(numpy_helper.to_array(model.graph.initializer[-1]),np.float32(.125))
        self.assertEqual(model.graph.initializer[-1].name,SCALE)
        onnx.checker.check_model(model)
        options=ort.SessionOptions()
        options.intra_op_num_threads=2
        options.log_severity_level=3
        session=ort.InferenceSession(model.SerializeToString(),options,providers=['CPUExecutionProvider'])
        for frames in [1,5]:
            x=np.random.default_rng(23).normal(0,.3,(1,frames,1024)).astype(np.float32)
            actual,=session.run(None,{'x':x})
            expected=reference(x,biases)
            self.assertEqual(actual.shape,expected.shape)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-6)

    def test_changed_topology_fails_before_any_candidate_mutation(self):
        baseline,_=fixture()
        for corruption in ['missing_layer','missing_bias','bias_bypass','head_order','masked_scores','already_fused','duplicate_named']:
            with self.subTest(corruption=corruption):
                model=copy.deepcopy(baseline)
                nodes={n.name:n for n in model.graph.node}
                prefix='/model/encoder/layers.23/self_attn/'
                if corruption=='missing_layer':
                    model.graph.node.remove(nodes[prefix+'Reshape_9'])
                elif corruption=='missing_bias':
                    target=next(p for p in model.graph.initializer if p.name=='model.encoder.layers.23.self_attn.q_proj.bias')
                    model.graph.initializer.remove(target)
                elif corruption=='bias_bypass':
                    nodes[prefix+'Reshape_2'].input[0]=prefix+'q_proj/MatMul_output_0'
                elif corruption=='head_order':
                    nodes[prefix+'sdpa/Transpose'].attribute[0].ints[:]=[0,1,2,3]
                elif corruption=='masked_scores':
                    nodes[prefix+'sdpa/Softmax'].input[0]='unverified_masked_scores'
                elif corruption=='already_fused':
                    nodes[prefix+'sdpa/MatMul'].op_type='MultiHeadAttention'
                else:
                    model.graph.node.append(copy.deepcopy(nodes[prefix+'q_proj/Add']))
                before=model.SerializeToString()
                with self.assertRaises(ValueError):
                    fuse_attention(model)
                self.assertEqual(model.SerializeToString(),before)


if __name__=='__main__':
    unittest.main()
