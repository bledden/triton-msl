"""CPU source-program admission. No Metal emitter or old stride descriptor."""
import struct
from dataclasses import replace

from . import address
demand=address.demand

def fbits(value):
    # The walker returns a raw integer word for -inf, floats for finite values.
    if type(value) is int:
        demand(value in (0,0xff800000),'unsupported raw float word')
        return value
    demand(type(value) is float,'float constant required')
    return struct.unpack('<I',struct.pack('<f',value))[0]

def node(op,shape,*parts): return (op,tuple(shape),*parts)
def c(bits,shape=()): return node('const',shape,bits)
def arg(i): return node('arg',(),i)
def binary(op,a,b):
    demand(a[1]==b[1],'binary expression shape')
    return node(op,a[1],a,b)
def broad(a,shape,axes):
    return node('broadcast',shape,a,tuple(axes))
def splat(a,shape): return broad(a,shape,())
def vec(a): return broad(a,(32,32),(0,))
def dot(a,b,acc): return node('dot',(32,32),a,b,acc,'ieee-source',0)

def descriptor(records):
    """Only complete records are consumed: no hand-written coefficient table."""
    result=[]
    for memory,ptr,record in records:
        divisions=[]
        def scan(expr):
            if expr[0]=='divsi': divisions.append(expr)
            for child in expr[1:]:
                if isinstance(child,tuple): scan(child)
        for link in record.links: scan(link)
        if record.bool_byte_cast:
            demand(len(divisions)==1 and divisions[0][2][0]=='arg','mask division record missing')
        else: demand(not divisions,'unexpected head division')
        result.append({'memory':memory,'pointer':ptr,'record':record,'head_division':tuple(divisions)})
    return tuple(result)

class Semantic:
    def __init__(self,lowerer,records):
        self.lowerer=lowerer; self.graph=lowerer.graph
        self.ap=address.Proof(lowerer).prepare()
        self.ops=self.ap.ops;self.args=self.ap.args
        self.mem={m:r for m,p,r in records}
        self.desc=descriptor(records)
        self.loop=self.ap.loop
        self.bids=self.loop.attrs['block_arg_ids'][1:]
        self.rids=self.loop.result_ids
        demand(len(self.rids)==3,'three loop results')
        self.cache={};self.active=set();self.used=set();self.locals=[]

    def facts(self,vid): return self.ap.facts(vid)
    def clean(self,op):
        attrs=self.ap.attrs(op,op.id)
        demand(all(v in self.ap.available[op.id] for v in op.operand_ids),'semantic operand does not dominate')
        return attrs

    def memdesc(self,vid):
        """Strict native-raw parser for the two observed shared tile layouts.

        The installed generic type parser calls these unknown. Bind ownership
        with the same checker on a temporary derived type, retaining the raw
        spelling and restoring the original record immediately.
        """
        meta=self.graph.result_meta[vid]
        op=self.ops[vid]
        raw=meta.type.raw
        prefix='!ttg.memdesc<32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = '
        allowed={prefix+order+'}>, #ttg.shared_memory>' for order in ('[1, 0]','[0, 1]')}
        demand(raw in allowed and op.op=='ttg.local_alloc','unproved native shared descriptor')
        derived=replace(meta.type,kind='memdesc',elem='f32',width=32,shape=(32,32),is_tensor=True,unknown_reason=None)
        self.graph.result_meta[vid]=replace(meta,type=derived)
        try: self.lowerer._native_value_facts(vid,op_name='shared allocation ownership')
        finally: self.graph.result_meta[vid]=meta
        self.locals.append(raw)

    def transport(self,op,attrs,child,shape):
        source=child[1]
        if op.op=='ttg.convert_layout':
            demand(not attrs and source==shape,'layout changes shape'); return child
        if op.op=='tt.splat':
            demand(not attrs and source==() and shape,'scalar splat'); return splat(child,shape)
        if op.op=='tt.expand_dims':
            axis=attrs.get('axis')
            demand(set(attrs)=={'axis'} and type(axis) is int and 0<=axis<=len(source) and shape==source[:axis]+(1,)+source[axis:],'semantic expansion')
            axes=tuple(i if i<axis else i+1 for i in range(len(source)))
        else:
            demand(op.op=='tt.broadcast' and not attrs and len(source)==len(shape) and all(a==b or a==1 for a,b in zip(source,shape)),'semantic broadcast')
            axes=tuple(range(len(source)))
        if child[0]=='broadcast':
            original=child[2]; oldaxes=child[3]
            axes=tuple(axes[a] for a in oldaxes)
            return broad(original,shape,axes)
        return broad(child,shape,axes)

    def value(self,vid):
        if vid in self.cache:return self.cache[vid]
        demand(vid not in self.active and len(self.active)<96 and len(self.used)<512,'bounded semantic proof')
        self.active.add(vid);self.used.add(vid)
        try:
            f=self.facts(vid); shape=f.shape
            if vid in self.args:
                demand(f.kind=='float' and f.elem=='f32' and not shape,'f32 scalar root')
                result=arg(self.args[vid].index)
            elif vid in self.bids:
                result=node('carry',shape,self.bids.index(vid))
            elif vid in self.rids:
                result=node('final',shape,self.rids.index(vid))
            else:
                op=self.ops[vid];attrs=self.clean(op);ids=op.operand_ids
                if op.op=='arith.constant':
                    demand(not ids and set(attrs)=={'value'},'constant attributes')
                    if f.elem=='f32': result=c(fbits(attrs['value']),shape)
                    else:
                        demand(f.elem=='i8' and attrs['value']==0,'only mask zero integer constant')
                        result=node('byte-zero',shape)
                elif op.op=='tt.load':
                    record=self.mem.get(vid)
                    demand(record is not None and len(ids)==2 and f.elem==record.element and shape==record.shape and attrs=={'isVolatile':False},'source masked load without invented other')
                    # Memory attributes are not arithmetic. Preserve them verbatim
                    # in receipt; memory masks are independently proved below.
                    result=node('load',shape,record.root)
                elif op.op=='ttg.local_load':
                    demand(len(ids)==1 and not attrs,'plain shared load')
                    alloc=self.ops[ids[0]];self.memdesc(alloc.id)
                    aattrs=self.clean(alloc)
                    demand(len(alloc.operand_ids)==1 and not aattrs and not alloc.region_ops,'plain initialized shared allocation')
                    users=[o for o in self.ops.values() if alloc.id in o.operand_ids]
                    demand(all(o.op=='ttg.local_load' and o.operand_ids==[alloc.id] for o in users),'shared descriptor has other users')
                    result=self.value(alloc.operand_ids[0])
                    demand(result[1]==shape==(32,32) and f.elem=='f32','shared shape/element')
                    self.used.add(alloc.id)
                elif op.op in ('tt.splat','tt.broadcast','tt.expand_dims','ttg.convert_layout'):
                    demand(len(ids)==1 and self.facts(ids[0]).elem==f.elem,'same-element semantic transport')
                    result=self.transport(op,attrs,self.value(ids[0]),shape)
                elif op.op in ('arith.addf','arith.subf','arith.mulf','arith.divf','arith.maxnumf'):
                    demand(len(ids)==2 and not attrs and f.elem=='f32' and all(self.facts(v).elem=='f32' and self.facts(v).shape==shape for v in ids),'plain f32 binary operation')
                    result=binary(op.op,self.value(ids[0]),self.value(ids[1]))
                elif op.op in ('math.exp2','math.log'):
                    demand(len(ids)==1 and not attrs and f.elem=='f32' and self.facts(ids[0]).shape==shape,'f32 transcendental')
                    result=node(op.op,shape,self.value(ids[0]))
                elif op.op=='arith.select':
                    demand(len(ids)==3 and not attrs and f.elem=='f32' and self.facts(ids[0]).elem=='i1' and all(self.facts(v).shape==shape for v in ids),'typed select')
                    result=node('select',shape,*(self.value(v) for v in ids))
                elif op.op=='arith.cmpi':
                    demand(len(ids)==2 and f.elem=='i1','comparison shape')
                    if attrs=={'predicate':1,'predicate_name':'ne'} and self.facts(ids[0]).elem=='i8':
                        result=node('byte-ne',shape,self.value(ids[0]),self.value(ids[1]))
                    else:
                        demand(attrs=={'predicate':2,'predicate_name':'slt'},'only signed source bounds')
                        coords=tuple(('lane',i) for i in range(len(shape)))
                        self.ap.visits=0
                        result=node('slt',shape,*(self.ap.integer(v,coords) for v in ids))
                elif op.op=='arith.andi':
                    demand(len(ids)==2 and not attrs and f.elem=='i1' and all(self.facts(v).shape==shape for v in ids),'bool and')
                    result=binary('and',self.value(ids[0]),self.value(ids[1]))
                elif op.op=='tt.dot':
                    demand(len(ids)==3 and attrs=={'maxNumImpreciseAcc':0} and f.elem=='f32' and shape==(32,32),'source dot attributes')
                    demand(all(self.facts(v).elem=='f32' and self.facts(v).shape==(32,32) for v in ids),'source dot types')
                    result=dot(*(self.value(v) for v in ids))
                elif op.op=='tt.reduce':
                    demand(len(ids)==1 and f.elem=='f32' and shape==(32,) and self.facts(ids[0]).shape==(32,32),'source reduction shapes')
                    demand(set(attrs)=={'axis','_parent_id','block_arg_ids','return_ids'} and attrs['axis']==1,'source reduction axis')
                    bids=attrs['block_arg_ids'];body=op.region_ops or []
                    demand(len(bids)==2 and len(body)==1,'reduction region grammar')
                    combiner=body[0];self.clean(combiner)
                    demand(combiner.op in ('arith.addf','arith.maxnumf') and combiner.operand_ids==bids and attrs['return_ids']==[combiner.id],'exact returned combiner')
                    demand(not self.ap.attrs(combiner,combiner.id) and all(self.facts(v).elem=='f32' and self.facts(v).shape==() for v in bids+[combiner.id]),'reduction scalar types')
                    result=node('reduce-'+combiner.op,shape,self.value(ids[0]));self.used.update(bids+[combiner.id])
                else: raise address.Refusal('unsupported semantic value '+op.op)
            self.cache[vid]=result
            return result
        finally:self.active.remove(vid)

    def prove(self):
        allowed=set('arith.constant arith.addi arith.muli arith.divsi arith.andi arith.cmpi arith.select arith.mulf arith.subf arith.addf arith.divf arith.maxnumf tt.addptr tt.get_program_id tt.make_range tt.broadcast tt.expand_dims tt.splat tt.bitcast tt.load tt.store tt.dot tt.reduce tt.return ttg.convert_layout ttg.local_alloc ttg.local_load scf.for scf.yield math.exp2 math.log'.split())
        census={}
        for op in address.uniform.walk(self.graph.ops):
            demand(op.op in allowed,'operation outside finite source grammar: '+op.op)
            demand(all(v in self.ap.available[op.id] for v in op.operand_ids),
                   'program operand does not dominate use')
            census[op.op]=census.get(op.op,0)+1
        demand(all(census.get(k,0)==n for k,n in {'tt.dot':2,'tt.reduce':2,'tt.load':5,'tt.store':2,'scf.for':1,'scf.yield':1,'ttg.local_alloc':4,'ttg.local_load':4}.items()),'source operation population')
        # This source-binding does not stand in for a general precision enum
        # decoder: exact reporter AST and the two dot nodes are bound by runner.
        memory={r.root:(self.ops[m],r) for m,r in self.mem.items()}
        n=('arg',9); lane=('lane',0);j=address.add(lane,address.mul(('pid',0),address.const(32)))
        k=address.add(lane,('iv',))
        jm=node('slt',(32,),j,n);km=node('slt',(32,),k,n)
        jmat=vec(jm);kmat=broad(km,(32,32),(1,));vmat=vec(km)
        both=binary('and',jmat,kmat)
        masks={2:jmat,3:kmat,4:vmat,5:both,6:km}
        for role,wanted in masks.items():
            op,r=memory[role]
            demand(self.value(op.operand_ids[1])==wanted,'load boundary mask differs for role '+str(role))
        load=lambda r:node('load',memory[r][1].shape,r)
        inv=fbits(1.4426950408889634);ln=fbits(0.6931471824645996)
        scale=binary('arith.mulf',arg(7),c(inv))
        q=binary('arith.mulf',load(2),splat(scale,(32,32)))
        bias=binary('arith.mulf',load(5),c(inv,(32,32)))
        score=dot(q,load(3),bias)
        masked=broad(node('byte-ne',(32,),load(6),node('byte-zero',(32,))),(32,32),(1,))
        sentinel=splat(arg(8),(32,32))
        score=node('select',(32,32),both,node('select',(32,32),masked,sentinel,score),sentinel)
        m=node('carry',(32,),0);l=node('carry',(32,),1);a=node('carry',(32,32),2)
        maximum=binary('arith.maxnumf',m,node('reduce-arith.maxnumf',(32,),score))
        p=node('math.exp2',(32,32),binary('arith.subf',score,vec(maximum)))
        alpha=node('math.exp2',(32,),binary('arith.subf',m,maximum))
        denominator=binary('arith.addf',binary('arith.mulf',l,alpha),node('reduce-arith.addf',(32,),p))
        numerator=dot(p,load(4),binary('arith.mulf',a,vec(alpha)))
        yields=[o for o in self.loop.region_ops if o.op=='scf.yield']
        demand(len(yields)==1 and len(yields[0].operand_ids)==3,'exact loop yield population')
        actual=tuple(self.value(v) for v in yields[0].operand_ids)
        demand(actual==(maximum,denominator,numerator),'source recurrence differs')
        demand(tuple(self.value(v) for v in self.loop.operand_ids[3:])==(c(0xff800000,(32,)),c(0,(32,)),c(0,(32,32))),'source initial state differs')
        fm=node('final',(32,),0);fl=node('final',(32,),1);fa=node('final',(32,32),2)
        out=binary('arith.divf',fa,vec(fl))
        lse=binary('arith.addf',binary('arith.mulf',fm,c(ln,(32,))),node('math.log',(32,),fl))
        padded=memory[1][1].shape==(32,1)
        want_lse=broad(lse,(32,1),(0,)) if padded else lse
        for role,wanted in ((0,out),(1,want_lse)):
            op,r=memory[role]
            demand(self.value(op.operand_ids[1])==wanted,'stored value differs for role '+str(role))
            demand(len(op.operand_ids)==(2 if padded else 3),'store predicate population')
            if not padded:demand(self.value(op.operand_ids[2])==(jmat if role==0 else jm),'store predicate differs')
        return {'padded':padded,'scale':scale,'bias':bias,'recurrence':actual,'out':out,'lse':want_lse,'constant_bits':{'inv_ln2':inv,'ln2':ln},'precision':'ieee from exact source; native enum not represented by installed walker','shared_descriptors':sorted(set(self.locals)),'addresses':self.desc,'operation_census':census}

def prove(lowerer):
    isolated,records,stamp=address.capture(lowerer)
    result=Semantic(isolated,records).prove()
    demand(address.digest(isolated.graph)==stamp,'semantic proof changed snapshot')
    return isolated,result
