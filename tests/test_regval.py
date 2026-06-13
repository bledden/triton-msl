from triton_metal.codegen.regval import RegVal, region_needs_arrays


class FakeOp:
    def __init__(self, op, operand_ids=(), region_ops=None, id=None):
        self.op = op; self.operand_ids = list(operand_ids)
        self.region_ops = region_ops; self.id = id


def test_regval_scalar_defaults():
    rv = RegVal(name="v0", n_elems=1, ty="float")
    assert rv.form == "scalar" and rv.is_scalar


def test_regval_array_form():
    rv = RegVal(name="v0", n_elems=4, ty="float", form="array")
    assert not rv.is_scalar and rv.n_elems == 4


def test_region_needs_arrays_straightline_false():
    ops = [FakeOp("tt.load", id=1), FakeOp("arith.addf", id=2), FakeOp("tt.store")]
    assert region_needs_arrays(ops, multi_elem_ids={1, 2}) is False


def test_region_needs_arrays_data_dependent_for_true():
    body = [FakeOp("tt.load"), FakeOp("arith.addf", operand_ids=[7])]
    ops = [FakeOp("scf.for", operand_ids=[7], region_ops=body)]
    assert region_needs_arrays(ops, multi_elem_ids={7}) is True


def test_region_needs_arrays_for_without_multielem_false():
    body = [FakeOp("arith.addi")]
    ops = [FakeOp("scf.for", region_ops=body)]
    assert region_needs_arrays(ops, multi_elem_ids=set()) is False
