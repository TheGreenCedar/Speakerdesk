"""Pinned ReDimNet2 MIL research adapter. No MLX import in parse/preflight.

CPU work is limited to artifact/literal/shape bookkeeping and host data copies.
Every neural primitive is explicitly dispatched to one MLX GPU stream. There is
no CPU retry, generic interpreter fallback, quantization or calibration rewrite.
"""
import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import struct
import numpy as np

EXPECTED = {
    'model.mil': '8b138359651f98bcd876eab2377ed9a5ddaf5f726745775ce1670ac5103fc79d',
    'weights/weight.bin': '780eb60151995b33d6a967deac32b9fe22ebd119e8288417f6d51ac7b856bcfc',
}
DTYPES = {'fp16': '<f2', 'fp32': '<f4', 'int32': '<i4', 'bool': '?'}
DECL = re.compile(r'^(tensor<[^>]+>|fp32|fp16|int32|bool|string) (\w+) = (\w+)\((.*?)\)\[name = string\("[^" ]+"\)(.*?)\];$')
KEYS = {
    'expand_dims': {'axes','x'}, 'squeeze': {'axes','x'},
    'reduce_mean': {'axes','keep_dims','x'}, 'reduce_sum': {'axes','keep_dims','x'},
    'reduce_l2_norm': {'axes','keep_dims','x'},
    'sub': {'x','y'}, 'add': {'x','y'}, 'mul': {'x','y'}, 'real_div': {'x','y'},
    'maximum': {'x','y'}, 'pow': {'x','y'},
    'square': {'x'}, 'sqrt': {'x'}, 'relu': {'x'}, 'tanh': {'x'},
    'pad': {'constant_val','mode','pad','x'},
    'conv': {'dilations','groups','pad','pad_type','strides','weight','x'},
    'clip': {'alpha','beta','x'}, 'log': {'epsilon','x'}, 'cast': {'dtype','x'},
    'slice_by_index': {'begin','end','end_mask','x'},
    'batch_norm': {'beta','epsilon','gamma','mean','variance','x'},
    'transpose': {'perm','x'}, 'reshape': {'shape','x'},
    'concat': {'axis','interleave','values'}, 'gelu': {'mode','x'},
    'linear': {'bias','weight','x'}, 'matmul': {'transpose_x','transpose_y','x','y'},
    'softmax': {'axis','x'}, 'layer_norm': {'axes','beta','epsilon','gamma','x'},
    'upsample_nearest_neighbor': {'scale_factor_height','scale_factor_width','x'},
    'tile': {'reps','x'},
}
DATA_KEYS = {'x','y','weight','bias','beta','gamma','mean','variance','values','alpha','epsilon'}

def split_top(text):
    depth = 0; start = 0; result = []
    for i,c in enumerate(text):
        if c in '([': depth += 1
        elif c in ')]': depth -= 1
        elif c == ',' and depth == 0: result.append(text[start:i].strip()); start = i+1
    if depth != 0: raise ValueError('Unbalanced MIL argument')
    if text.strip(): result.append(text[start:].strip())
    return result

def tensor_type(text):
    m = re.fullmatch(r'tensor<(\w+), \[([\d, ]+)\]>',text)
    if m: return m[1],tuple(int(v.strip()) for v in m[2].split(','))
    if text in (*DTYPES,'string'): return text,()
    raise ValueError('Unsupported MIL type: '+text)

def literal(text,dtype,shape):
    prefix = text.find('(')
    if prefix < 0 or not text.endswith(')'): raise ValueError('Invalid MIL literal')
    content = text[prefix+1:-1]
    content = re.sub(r'-?0x[\da-f.]+p[+-]?\d+',lambda m:repr(float.fromhex(m[0])),content)
    content = re.sub(r'\btrue\b','True',content); content = re.sub(r'\bfalse\b','False',content)
    if content in ('inf','-inf'): value = float(content)
    else: value = ast.literal_eval(content)
    if dtype == 'string':
        if not isinstance(value,str): raise ValueError('String literal required')
        return value
    array = np.asarray(value,dtype=DTYPES[dtype])
    if array.shape != shape: raise ValueError('MIL literal shape mismatch')
    return array

@dataclass
class Node:
    name: str
    op: str
    dtype: str
    shape: tuple
    args: dict
    value: object = None

class Graph:
    def __init__(self,root):
        self.root = Path(root)
        raw = {}
        for name,digest in EXPECTED.items():
            path = self.root/name
            if path.is_symlink() or not path.is_file(): raise ValueError('Missing/linked pinned artifact')
            raw[name] = path.read_bytes()
            if hashlib.sha256(raw[name]).hexdigest() != digest: raise ValueError('Pinned artifact hash mismatch')
        blob = raw['weights/weight.bin']; count,version,*reserved = struct.unpack_from('<II7Q',blob)
        if version != 2 or any(reserved): raise ValueError('Unsupported weight storage')
        self.nodes = []; self.byname = {'audio': Node('audio','input','fp32',(1,96000),{})}
        spans = []; external = 0
        for rawline in raw['model.mil'].decode().splitlines():
            line = rawline.strip()
            if ' = ' not in line or line.startswith('[buildInfo'): continue
            match = DECL.fullmatch(line)
            if not match: raise ValueError('Unsupported MIL declaration')
            typ,name,op,args,attrs = match.groups(); dtype,shape = tensor_type(typ)
            if name in self.byname: raise ValueError('Duplicate MIL declaration')
            arguments = {}
            for item in split_top(args):
                key,ref = item.split(' = ',1)
                refs = tuple(re.findall(r'\b[A-Za-z_]\w*\b',ref))
                if not refs or any(r not in self.byname for r in refs): raise ValueError('Invalid/forward MIL reference')
                arguments[key] = refs if ref.startswith('(') else refs[0]
            node = Node(name,op,dtype,shape,arguments)
            if op == 'const':
                if not attrs.startswith(', val = '): raise ValueError('Unknown constant attributes')
                val = attrs[len(', val = '):]
                if 'BLOBFILE' in val:
                    match = re.fullmatch(r'tensor<[^>]+>\(BLOBFILE\(path = string\("@model_path/weights/weight.bin"\), offset = uint64\((\d+)\)\)\)',val)
                    if not match or dtype not in ('fp16','fp32'): raise ValueError('Unsupported weight reference')
                    offset = int(match[1]); marker,code,size,data,*rest = struct.unpack_from('<II7Q',blob,offset)
                    if offset%64 or marker != 0xDEADBEEF or any(rest) or code != {'fp16':1,'fp32':2}[dtype]: raise ValueError('Weight metadata mismatch')
                    if size != math.prod(shape)*np.dtype(DTYPES[dtype]).itemsize or data%64 or data<offset+64 or data+size>len(blob): raise ValueError('Weight bounds/shape mismatch')
                    node.value = np.frombuffer(blob,dtype=DTYPES[dtype],count=math.prod(shape),offset=data).reshape(shape)
                    spans.append((offset,data+size)); external += 1
                else: node.value = literal(val,dtype,shape)
            elif attrs or op not in KEYS or (set(arguments)-({'bias'} if op=='conv' else set())) != KEYS[op]: raise ValueError('Unsupported MIL operator/attributes: '+op)
            self.nodes.append(node); self.byname[name] = node
        spans.sort()
        if external != count or count != 627 or any(a[1]>b[0] for a,b in zip(spans,spans[1:])): raise ValueError('Weight span inventory mismatch')
        if len(self.nodes) != 2685 or self.byname.get('embedding',None).shape != (1,192): raise ValueError('Graph inventory mismatch')
        self.preflight()

    def const(self,node,key):
        ref = node.args[key]; n = self.byname[ref]
        if n.op != 'const': raise ValueError('Dynamic operator attribute: '+key)
        return n.value.tolist() if isinstance(n.value,np.ndarray) else n.value

    def shape(self,node,key='x'):
        return self.byname[node.args[key]].shape

    def axes(self,node,rank):
        axes = tuple(int(v)%rank for v in self.const(node,'axes'))
        if len(set(axes)) != len(axes): raise ValueError('Duplicate axis')
        return axes

    def conv_padding(self,node):
        shape = self.shape(node); weight = self.shape(node,'weight'); n = len(shape)-2
        stride = self.const(node,'strides'); dilation = self.const(node,'dilations'); mode = self.const(node,'pad_type')
        if n not in (1,2) or len(weight)!=len(shape) or len(stride)!=n or len(dilation)!=n: raise ValueError('Unsupported convolution rank')
        if mode == 'valid': return [(0,0)]*n
        if mode == 'custom':
            values = self.const(node,'pad')
            if len(values) != 2*n: raise ValueError('Convolution padding rank')
            return list(zip(values[::2],values[1::2]))
        if mode == 'same':
            values = [max(0,(math.ceil(d/s)-1)*s+(k-1)*a+1-d) for d,s,k,a in zip(shape[2:],stride,weight[2:],dilation)]
            return [(p//2,p-p//2) for p in values]
        raise ValueError('Unsupported convolution padding')

    def preflight(self):
        for node in self.nodes:
            if node.op == 'const': continue
            if node.dtype not in ('fp16','fp32'): raise ValueError('Unsupported neural dtype')
            op=node.op; shape=self.shape(node) if 'x' in node.args else None
            result=shape
            # Attribute and shape checks use Python integers only, never arrays of activations.
            for key,ref in node.args.items():
                if key not in DATA_KEYS and key not in ('beta',): self.const(node,key)
            if op in ('add','sub','mul','real_div','pow','maximum'):
                result=np.broadcast_shapes(shape,self.shape(node,'y'))
            elif op=='expand_dims':
                rank=len(shape)+len(self.const(node,'axes')); axes=sorted(int(v)%rank for v in self.const(node,'axes')); result=list(shape)
                for axis in axes: result.insert(axis,1)
                result=tuple(result)
            elif op=='squeeze':
                axes=self.axes(node,len(shape))
                if any(shape[a]!=1 for a in axes): raise ValueError('Squeezed axis is not one')
                result=tuple(v for i,v in enumerate(shape) if i not in axes)
            elif op.startswith('reduce_'):
                axes=self.axes(node,len(shape)); keep=self.const(node,'keep_dims')
                result=tuple(1 if i in axes else v for i,v in enumerate(shape)) if keep else tuple(v for i,v in enumerate(shape) if i not in axes)
            elif op=='pad':
                if self.const(node,'mode')!='reflect' or self.const(node,'constant_val')!=0: raise ValueError('Unsupported waveform padding')
                pad=self.const(node,'pad')
                if len(pad)!=len(shape)*2 or any(p<0 for p in pad): raise ValueError('Padding rank')
                result=tuple(v+pad[2*i]+pad[2*i+1] for i,v in enumerate(shape))
            elif op=='conv':
                weight=self.shape(node,'weight'); groups=self.const(node,'groups'); strides=self.const(node,'strides'); dilations=self.const(node,'dilations'); pads=self.conv_padding(node)
                if groups<1 or shape[1]%groups or weight[0]%groups or weight[1]*groups!=shape[1] or any(v<1 for v in (*strides,*dilations)): raise ValueError('Invalid convolution groups/stride')
                result=(shape[0],weight[0],*((d+p[0]+p[1]-(k-1)*a-1)//s+1 for d,p,k,a,s in zip(shape[2:],pads,weight[2:],dilations,strides)))
                if 'bias' in node.args and self.shape(node,'bias')!=(weight[0],):raise ValueError('Convolution bias shape')
            elif op=='cast':
                if self.const(node,'dtype')!=node.dtype: raise ValueError('Cast dtype mismatch')
            elif op=='slice_by_index':
                begin=self.const(node,'begin');end=self.const(node,'end');mask=self.const(node,'end_mask')
                if not len(begin)==len(end)==len(mask)==len(shape):raise ValueError('Slice rank')
                result=tuple((v if mask[i] else min(end[i],v))-begin[i] for i,v in enumerate(shape))
            elif op=='transpose':
                perm=self.const(node,'perm')
                if sorted(perm)!=list(range(len(shape))):raise ValueError('Transpose axes')
                result=tuple(shape[i] for i in perm)
            elif op=='reshape':
                target=self.const(node,'shape');unknown=[i for i,v in enumerate(target) if v==-1]
                if len(unknown)>1 or any(v==0 or v<-1 for v in target):raise ValueError('Reshape shape')
                if unknown:target[unknown[0]]=math.prod(shape)//math.prod(v for v in target if v!=-1)
                if math.prod(target)!=math.prod(shape):raise ValueError('Reshape size')
                result=tuple(target)
            elif op=='concat':
                if self.const(node,'interleave') is not False:raise ValueError('Interleaved concat unsupported')
                refs=node.args['values'];refs=refs if isinstance(refs,tuple) else (refs,)
                shapes=[self.byname[r].shape for r in refs];axis=self.const(node,'axis')%len(shapes[0]);result=list(shapes[0]);result[axis]=sum(s[axis] for s in shapes)
                if any(tuple(v for i,v in enumerate(s) if i!=axis)!=tuple(v for i,v in enumerate(shapes[0]) if i!=axis) for s in shapes):raise ValueError('Concat shape')
                result=tuple(result)
            elif op=='gelu':
                if self.const(node,'mode') not in ('EXACT','TANH_APPROXIMATION'):raise ValueError('GELU mode')
            elif op=='linear':
                weight=self.shape(node,'weight')
                if len(weight)!=2 or shape[-1]!=weight[1] or self.shape(node,'bias')!=(weight[0],):raise ValueError('Linear shape')
                result=(*shape[:-1],weight[0])
            elif op=='matmul':
                x=list(shape);y=list(self.shape(node,'y'))
                if self.const(node,'transpose_x'):x[-1],x[-2]=x[-2],x[-1]
                if self.const(node,'transpose_y'):y[-1],y[-2]=y[-2],y[-1]
                if x[-1]!=y[-2]:raise ValueError('Matmul shape')
                result=(*np.broadcast_shapes(tuple(x[:-2]),tuple(y[:-2])),x[-2],y[-1])
            elif op=='batch_norm':
                if any(self.shape(node,k)!=(shape[1],) for k in ('beta','gamma','mean','variance')) or self.const(node,'epsilon')<=0:raise ValueError('Batch norm shape/epsilon')
            elif op=='layer_norm':
                axes=self.axes(node,len(shape));normshape=tuple(shape[i] for i in axes)
                if self.shape(node,'gamma')!=normshape or self.shape(node,'beta')!=normshape or self.const(node,'epsilon')<=0:raise ValueError('Layer norm shape/epsilon')
            elif op=='upsample_nearest_neighbor':
                h=self.const(node,'scale_factor_height');w=self.const(node,'scale_factor_width')
                if len(shape)!=4 or type(h)!=int or type(w)!=int or h<1 or w<1:raise ValueError('Resize scale')
                result=(*shape[:2],shape[2]*h,shape[3]*w)
            elif op=='tile':
                reps=self.const(node,'reps')
                if len(reps)!=len(shape) or any(v<1 for v in reps):raise ValueError('Tile repetitions')
                result=tuple(a*b for a,b in zip(shape,reps))
            elif op=='softmax':
                axis=self.const(node,'axis')
                if not -len(shape)<=axis<len(shape):raise ValueError('Softmax axis')
            if tuple(result)!=node.shape:raise ValueError(f'Shape inference differs at {node.name}: {result} != {node.shape}')

    def manifest(self):
        counts=Counter(n.op for n in self.nodes)
        return {'scope':'Pinned complete operator/attribute/integer-shape preflight; zero neural calls',
                'files':EXPECTED,'ops':dict(counts),'nonconstant_ops':sum(v for k,v in counts.items() if k!='const'),
                'declared_dtypes':dict(Counter(n.dtype for n in self.nodes if n.op!='const')),
                'input_shape':[1,96000],'output_shape':[1,192],'external_tensors':627}

class GPUProgram:
    def __init__(self,graph,mx):
        self.graph,self.mx=graph,mx
        if not mx.metal.is_available():raise ValueError('Metal GPU unavailable; no CPU fallback')
        self.stream=mx.new_stream(mx.gpu)
        self.constants={}
        self.dtypes={'fp16':mx.float16,'fp32':mx.float32,'int32':mx.int32,'bool':mx.bool_}

    def call(self,name,*args,**kwargs):
        return getattr(self.mx,name)(*args,stream=self.stream,**kwargs)

    def const_array(self,name):
        if name not in self.constants:
            node=self.graph.byname[name]
            if node.op!='const' or node.dtype=='string':raise ValueError('Numeric constant required')
            self.constants[name]=self.mx.array(node.value,dtype=self.dtypes[node.dtype])
        return self.constants[name]

    def forward(self,audio,progress=lambda node:None):
        mx=self.mx;g=self.graph;c=self.call
        audio=np.asarray(audio)
        if audio.shape!=(1,96000) or audio.dtype!=np.float32 or not np.isfinite(audio).all():raise ValueError('Prepared synthetic input shape/dtype')
        values={'audio':mx.array(audio)}
        uses=Counter(ref for n in g.nodes if n.op!='const' for refs in n.args.values() for ref in (refs if isinstance(refs,tuple) else (refs,)))
        def resolve(ref):
            if isinstance(ref,tuple):return tuple(resolve(r) for r in ref)
            return self.const_array(ref) if g.byname[ref].op=='const' else values[ref]
        with mx.stream(self.stream):
            for node in g.nodes:
                if node.op=='const':continue
                op=node.op; args={k:resolve(v) for k,v in node.args.items() if k in DATA_KEYS or k=='beta'}
                x=args.get('x');attr=lambda k:g.const(node,k)
                if op in ('add','sub','mul','real_div','pow','maximum'):
                    y=c({'add':'add','sub':'subtract','mul':'multiply','real_div':'divide','pow':'power','maximum':'maximum'}[op],x,args['y'])
                elif op in ('square','sqrt','tanh'):y=c(op,x)
                elif op=='relu':y=c('maximum',x,0)
                elif op=='log':y=c('log',c('add',x,args['epsilon']))
                elif op=='clip':y=c('clip',x,args['alpha'],args['beta'])
                elif op=='cast':y=x.astype(self.dtypes[attr('dtype')],stream=self.stream)
                elif op=='pad':y=c('pad',x,list(zip(attr('pad')[::2],attr('pad')[1::2])),mode=attr('mode'))
                elif op=='conv':
                    rank=len(g.shape(node));perm=(0,*range(2,rank),1)
                    inp=c('transpose',x,perm);weight=c('transpose',args['weight'],perm)
                    pads=g.conv_padding(node)
                    y=c('conv_general',inp,weight,stride=attr('strides'),padding=([p[0] for p in pads],[p[1] for p in pads]),kernel_dilation=attr('dilations'),groups=attr('groups'))
                    y=c('transpose',y,(0,rank-1,*range(1,rank-1)))
                    if 'bias' in args:y=c('add',y,c('reshape',args['bias'],(1,node.shape[1],*(1 for _ in node.shape[2:]))))
                elif op=='expand_dims':
                    y=x;rank=len(node.shape)
                    for axis in sorted(int(v)%rank for v in attr('axes')):y=c('expand_dims',y,axis)
                elif op=='squeeze':y=c('squeeze',x,axis=tuple(attr('axes')))
                elif op.startswith('reduce_'):
                    kind={'reduce_mean':'mean','reduce_sum':'sum','reduce_l2_norm':'sum'}[op]
                    # MLX reductions accumulate half inputs in float32 internally.
                    y=c(kind,c('square',x) if op=='reduce_l2_norm' else x,axis=tuple(attr('axes')),keepdims=attr('keep_dims'))
                    if op=='reduce_l2_norm':y=c('sqrt',y)
                elif op=='slice_by_index':
                    sl=tuple(slice(b,None if m else e) for b,e,m in zip(attr('begin'),attr('end'),attr('end_mask')))
                    y=x[sl] # Runs inside the explicit GPU stream context.
                elif op=='transpose':y=c('transpose',x,attr('perm'))
                elif op=='reshape':y=c('reshape',x,node.shape)
                elif op=='concat':
                    arr=args['values'];arr=arr if isinstance(arr,tuple) else (arr,)
                    y=c('concatenate',arr,axis=attr('axis'))
                elif op=='gelu':
                    work=x.astype(mx.float32,stream=self.stream)
                    if attr('mode')=='EXACT':activation=c('erf',c('multiply',work,1/math.sqrt(2)))
                    else:activation=c('tanh',c('multiply',math.sqrt(2/math.pi),c('add',work,c('multiply',.044715,c('power',work,3)))))
                    y=c('multiply',c('multiply',work,.5),c('add',activation,1))
                elif op=='linear':y=c('add',c('matmul',x,c('transpose',args['weight'],(1,0))),args['bias'])
                elif op=='matmul':
                    a=x;b=args['y']
                    if attr('transpose_x'):a=c('swapaxes',a,-1,-2)
                    if attr('transpose_y'):b=c('swapaxes',b,-1,-2)
                    y=c('matmul',a,b)
                elif op=='softmax':y=c('softmax',x,axis=attr('axis'),precise=True)
                elif op in ('batch_norm','layer_norm'):
                    work=x.astype(mx.float32,stream=self.stream)
                    if op=='batch_norm':
                        shape=(1,x.shape[1],*(1 for _ in x.shape[2:]));v=lambda k:c('reshape',args[k].astype(mx.float32,stream=self.stream),shape)
                        mean=v('mean');variance=v('variance');gamma=v('gamma');beta=v('beta')
                    else:
                        axes=tuple(attr('axes'));mean=c('mean',work,axis=axes,keepdims=True);variance=c('mean',c('square',c('subtract',work,mean)),axis=axes,keepdims=True)
                        gamma=args['gamma'].astype(mx.float32,stream=self.stream);beta=args['beta'].astype(mx.float32,stream=self.stream)
                    y=c('add',c('multiply',c('divide',c('subtract',work,mean),c('sqrt',c('add',variance,attr('epsilon')))),gamma),beta)
                elif op=='upsample_nearest_neighbor':
                    y=c('repeat',c('repeat',x,attr('scale_factor_height'),axis=2),attr('scale_factor_width'),axis=3)
                elif op=='tile':y=c('tile',x,attr('reps'))
                else:raise ValueError('Unknown GPU operation: '+op)
                y=y.astype(self.dtypes[node.dtype],stream=self.stream)
                if tuple(y.shape)!=node.shape or y.dtype!=self.dtypes[node.dtype]:raise ValueError('GPU result contract: '+node.name)
                values[node.name]=y
                # Evaluate each declaration boundary to enforce FP16 rounding and bound graph retention.
                mx.eval(y);mx.synchronize(self.stream);progress(node)
                for refs in node.args.values():
                    for ref in (refs if isinstance(refs,tuple) else (refs,)):
                        uses[ref]-=1
                        if uses[ref]==0:values.pop(ref,None)
            return values['embedding']
