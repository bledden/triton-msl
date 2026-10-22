"""Closed typed address integration, isolated from production dispatch.

Output is a tuple IR: it retains source operation order, i32 width, and each
pointer link. It is deliberately not the legacy coefficient-only descriptor.
"""
import copy
from dataclasses import dataclass
import hashlib
from . import uniform
Refusal=uniform.Refusal

def demand(ok, message):
    if not ok: raise Refusal(message)

def digest(graph):
    # Internal compiler dataclasses, not caller objects. Deepcopy and comparison
    # bracket capture; the captured graph remains private to this CPU operation.
    return hashlib.sha256(repr(graph).encode()).hexdigest()

def uexpr(expr):
    return (expr.op,expr.value) if not expr.children else (expr.op,*(uexpr(c) for c in expr.children))

def const(n): return ('const',n & 0xffffffff)
def mul(a,b): return ('mul',a,b)
def add(a,b): return ('add',a,b)

def expected():
    n=('arg',9); h=('arg',10); d=const(32)
    nn=mul(n,n); nd=mul(n,d); nnd=mul(nn,d)
    x,y,z=(('pid',i) for i in range(3))
    r,c=('lane',0),('lane',1)
    j=add(r,mul(x,d)); kr=add(r,('iv',)); kc=add(c,('iv',))
    base=(mul(z,nnd),mul(y,nd))
    return {
      0:((32,32),'f32',base+(mul(j,d),c)),
      1:((32,),'f32',(mul(z,nn),mul(y,n),j)),
      2:((32,32),'f32',base+(mul(j,d),c)),
      3:((32,32),'f32',base+(r,mul(kc,d))),
      4:((32,32),'f32',base+(mul(kr,d),c)),
      5:((32,32),'f32',(mul(z,nn),mul(j,n),kc)),
      6:((32,),'i8',(mul(('divsi',z,h),nn),mul(y,n),kr)),
    }

@dataclass(frozen=True)
class Address:
    root: int
    shape: tuple
    element: str
    links: tuple
    bool_byte_cast: bool=False

class Proof:
    def __init__(self, lowerer):
        self.lowerer=lowerer
        self.graph=lowerer.graph
        # Do not inherit a cache that predates graph validation.
        for attr in ('_native_value_owners','_native_owner_ids'):
            if hasattr(lowerer,attr): delattr(lowerer,attr)
        self.uniform=uniform.UniformProof(lowerer)
        self.ops=self.uniform.ops
        self.args=self.uniform.args
        self.epoch=digest(self.graph)
        self.visits=0
        self.active=set()
        self.loop=None
        self.iv=None
        self.cones={}
        self.available={}
        # Bind def-use scope as well as native ownership. Equal arithmetic from
        # a later op or sibling region must not become an admissible address.
        owned=set(self.args)|set(self.ops)
        def scopes(ops,available):
            available=set(available)
            for op in ops:
                self.available[op.id]=frozenset(available)
                bids=list((op.attrs or {}).get('block_arg_ids') or [])
                demand(len(bids)==len(set(bids)) and not (set(bids)&owned),'duplicate block argument owner')
                owned.update(bids)
                if op.region_ops: scopes(op.region_ops,available|set(bids))
                demand(not op.else_ops,'else region outside reporter address scope')
                available.update(op.result_ids or [op.id])
        scopes(self.graph.ops,set(self.args))

    def facts(self,vid):
        f=self.lowerer._native_value_facts(vid,op_name='typed reporter address')
        demand(type(f.shape) is tuple and all(type(x) is int and x>0 for x in f.shape),'unknown native shape')
        demand(f.is_tensor==bool(f.shape),'tensor/shape contradiction')
        return f

    def attrs(self,op,vid):
        attrs=dict(op.attrs or {})
        meta=self.graph.result_meta.get(vid)
        demand(meta is not None,'missing metadata')
        demand(attrs.get('_block_id',meta.owner_id)==meta.owner_id,'block ownership differs')
        order=attrs.pop('_walk_order',0)
        demand(type(order) is int and order>=0,'invalid operation order')
        attrs.pop('_block_id',None)
        return attrs

    def node(self,vid):
        op=self.ops.get(vid)
        demand(op is not None and op.id==vid and (not op.result_ids or op.result_ids==[vid]),'one result required')
        demand(not op.region_ops and not op.else_ops,'unexpected address region')
        demand(all(v in self.available[op.id] for v in op.operand_ids),'address operand does not dominate use')
        return op,self.attrs(op,vid)

    def transport(self,op,attrs,source,target,coords):
        demand(len(op.operand_ids)==1,'unary transport arity')
        a,b=source.shape,target.shape
        if op.op=='tt.splat':
            demand(a==() and b and not attrs,'invalid splat'); return ()
        if op.op=='ttg.convert_layout':
            demand(a==b and not attrs,'invalid layout conversion'); return coords
        if op.op=='tt.broadcast':
            demand(len(a)==len(b) and all(x==y or x==1 for x,y in zip(a,b)) and not attrs,'invalid broadcast')
            return tuple(const(0) if x==1 else c for x,c in zip(a,coords))
        axis=attrs.get('axis')
        demand(op.op=='tt.expand_dims' and set(attrs)=={'axis'} and type(axis) is int and 0<=axis<=len(a) and b==a[:axis]+(1,)+a[axis:],'invalid expand axis')
        return coords[:axis]+coords[axis+1:]

    def enter(self,vid,depth):
        self.visits+=1
        demand(self.visits<=2048 and depth<=96 and vid not in self.active,'address proof budget/cycle')
        self.active.add(vid)

    def integer(self,vid,coords,depth=0):
        self.enter(vid,depth)
        try:
            f=self.facts(vid)
            demand(f.kind=='integer' and f.elem=='i32' and f.width==32 and len(coords)==len(f.shape),'i32 offset required')
            if vid in self.args:
                expr=self.uniform._prove(vid)
                return uexpr(expr)
            if vid==self.iv:
                demand(f.shape==(),'scalar induction required')
                return ('iv',)
            op,attrs=self.node(vid)
            ids=op.operand_ids
            if op.op=='arith.constant':
                return uexpr(self.uniform._prove(vid))
            if op.op=='tt.get_program_id':
                demand(not ids and f.shape==() and set(attrs)=={'axis'} and type(attrs['axis']) is int and 0<=attrs['axis']<=2,'invalid program axis')
                return ('pid',attrs['axis'])
            if op.op=='tt.make_range':
                demand(not ids and f.shape==(32,) and attrs=={'start':0,'end':32},'unsupported lane range')
                return coords[0]
            if op.op in ('arith.addi','arith.muli','arith.divsi'):
                demand(len(ids)==2 and not attrs,'plain binary arithmetic required')
                demand(all(self.facts(v).shape==f.shape for v in ids),'arithmetic shape mismatch')
                return ({'arith.addi':'add','arith.muli':'mul','arith.divsi':'divsi'}[op.op],*(self.integer(v,coords,depth+1) for v in ids))
            if op.op in ('tt.splat','tt.broadcast','tt.expand_dims','ttg.convert_layout'):
                demand(len(ids)==1,'transport arity')
                sf=self.facts(ids[0])
                demand((sf.kind,sf.elem,sf.width)==('integer','i32',32),'transport element changed')
                mapped=self.transport(op,attrs,sf,f,coords)
                return self.integer(ids[0],mapped,depth+1)
            raise Refusal('unsupported offset operation: '+op.op)
        finally: self.active.remove(vid)

    def pointer(self,vid,coords,depth=0):
        self.enter(vid,depth)
        try:
            f=self.facts(vid)
            demand(f.kind=='pointer' and f.address_space==1 and len(coords)==len(f.shape),'global pointer required')
            arg=self.args.get(vid)
            if arg is not None:
                demand(arg.is_ptr and not f.shape and arg.index in range(7),'invalid pointer root')
                return arg.index,(),False
            op,attrs=self.node(vid)
            ids=op.operand_ids
            if op.op=='tt.addptr':
                demand(len(ids)==2 and not attrs,'plain addptr required')
                sf=self.facts(ids[0])
                demand(sf.kind=='pointer' and sf.shape==f.shape and sf.pointee==f.pointee and sf.address_space==f.address_space,'addptr pointer type changed')
                demand(self.facts(ids[1]).shape==f.shape,'offset shape mismatch')
                root,links,cast=self.pointer(ids[0],coords,depth+1)
                demand(not cast,'offset after bool conversion refused')
                return root,links+(self.integer(ids[1],coords,depth+1),),cast
            if op.op in ('tt.splat','tt.broadcast','tt.expand_dims','ttg.convert_layout'):
                demand(len(ids)==1,'pointer transport arity')
                sf=self.facts(ids[0])
                demand(sf.kind=='pointer' and sf.pointee==f.pointee and sf.address_space==f.address_space,'pointer transport type changed')
                return self.pointer(ids[0],self.transport(op,attrs,sf,f,coords),depth+1)
            if op.op=='tt.bitcast':
                demand(len(ids)==1 and not attrs,'one plain bitcast required')
                sf=self.facts(ids[0])
                demand(sf.kind=='pointer' and sf.shape==f.shape and sf.address_space==f.address_space and sf.pointee.elem=='i1' and f.pointee.elem=='i8','only bool-to-byte load bridge')
                root,links,cast=self.pointer(ids[0],coords,depth+1)
                demand(root==6 and not cast,'mask cast root/replay')
                return root,links,True
            raise Refusal('unsupported pointer operation: '+op.op)
        finally: self.active.remove(vid)

    def prepare(self):
        demand(not self.graph.called_funcs,'callee unsupported')
        demand(len(self.graph.args)==11,'reporter signature arity')
        for index,arg in enumerate(self.graph.args):
            f=self.facts(arg.id)
            want='i1' if index==6 else 'f32' if index<9 else 'i32'
            demand(arg.index==index and f.shape==() and arg.is_ptr==(index<7),'argument slot/type')
            demand(arg.elem_type==want,'argument declaration contradicts native type')
            demand((f.pointee.elem if index<7 and f.kind=='pointer' else f.elem)==want,'argument element')
        loops=[o for o in uniform.walk(self.graph.ops) if o.op=='scf.for']
        demand(len(loops)==1,'one source loop required')
        self.loop=loops[0]
        bids=self.loop.attrs.get('block_arg_ids',[])
        demand(len(self.loop.operand_ids)==6 and len(bids)==4,'source loop shape')
        for vid,wanted in zip(self.loop.operand_ids[:3],(const(0),('arg',9),const(32))):
            demand(uexpr(self.uniform._prove(vid))==wanted,'source induction bounds/step changed')
        self.iv=bids[0]
        f=self.facts(self.iv)
        demand(f.kind=='integer' and f.elem=='i32' and f.width==32 and f.shape==(),'source induction type')
        return self

    def prove_all(self):
        self.prepare()
        wanted=expected(); result=[]
        memory=[o for o in uniform.walk(self.graph.ops) if o.op in ('tt.load','tt.store')]
        demand(len(memory)==7,'complete seven-operation memory population')
        inner={o.id for o in uniform.walk(self.loop.region_ops)}
        seen=set()
        for op in memory:
            demand(op.operand_ids,'memory pointer missing')
            ptr=op.operand_ids[0]; facts=self.facts(ptr)
            demand(ptr in self.available[op.id],'memory pointer does not dominate use')
            self.visits=0
            root,links,cast=self.pointer(ptr,tuple(('lane',i) for i in range(len(facts.shape))))
            demand(root not in seen,'duplicate role'); seen.add(root)
            record=Address(root,facts.shape,facts.pointee.elem,links,cast)
            expected_shape,expected_element,expected_links=wanted[root]
            # Reporter PADDED stores LSE through an explicit trailing singleton;
            # the masked branch stores rank one. Preserve that shape in receipts,
            # and require exactly the same lane-0 links in either spelling.
            shape_ok=record.shape==expected_shape or (root==1 and record.shape==(32,1))
            demand(shape_ok and record.element==expected_element and record.links==expected_links,'complete address differs for role '+str(root))
            demand(cast==(root==6),'mask conversion required only for mask')
            demand(op.op==('tt.store' if root in (0,1) else 'tt.load'),'role access direction')
            demand((op.id in inner)==(root in (3,4,5,6)),'role loop placement')
            result.append((op.id,ptr,record))
        demand(seen==set(wanted),'missing pointer role')
        demand(digest(self.graph)==self.epoch,'graph mutated during proof')
        return tuple(result)

def capture(lowerer):
    before=digest(lowerer.graph)
    snapshot=copy.deepcopy(lowerer.graph)
    demand(digest(snapshot)==before==digest(lowerer.graph),'graph changed during snapshot')
    isolated=type(lowerer)(snapshot,copy.deepcopy(lowerer.options))
    result=Proof(isolated).prove_all()
    demand(digest(snapshot)==before==digest(lowerer.graph),'graph changed during integration')
    return isolated,result,before

def bits(expr,args,pids,lanes,iv):
    kind=expr[0]
    if kind=='const': return expr[1]
    if kind=='arg': return args[expr[1]] & 0xffffffff
    if kind=='pid': return pids[expr[1]] & 0xffffffff
    if kind=='lane': return lanes[expr[1]] & 0xffffffff
    if kind=='iv': return iv & 0xffffffff
    a=bits(expr[1],args,pids,lanes,iv); b=bits(expr[2],args,pids,lanes,iv)
    if kind=='add': return (a+b)&0xffffffff
    if kind=='mul': return (a*b)&0xffffffff
    demand(kind=='divsi','unknown offset expression')
    a=a if a<0x80000000 else a-(1<<32); b=b if b<0x80000000 else b-(1<<32)
    demand(b!=0 and (a,b)!=(-(1<<31),-1),'undefined signed division')
    return ((abs(a)//abs(b)) * (-1 if (a<0)!=(b<0) else 1)) & 0xffffffff

def offset(record,args,pids,lanes,iv):
    values=[bits(e,args,pids,lanes,iv) for e in record.links]
    return sum(v if v<0x80000000 else v-(1<<32) for v in values) & 0xffffffffffffffff
