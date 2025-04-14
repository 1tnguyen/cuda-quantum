# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np, pytest, random
from cudaq.ops import * # FIXME: module name
from op_utils import * # test helpers


@pytest.fixture(autouse=True)
def setup():
    random.seed(10)
    yield


def test_definitions():
    dims = {0: 2, 1: 3}
    assert np.allclose(number(1).to_matrix(dims), number_matrix(3))
    assert np.allclose(parity(1).to_matrix(dims), parity_matrix(3))
    assert np.allclose(position(1).to_matrix(dims), position_matrix(3))
    assert np.allclose(momentum(1).to_matrix(dims), momentum_matrix(3))
    assert np.allclose(squeeze(1).to_matrix(dims, squeezing = 0.5 + 1.2j, displacement = 0.5 + 1.2j), squeeze_matrix(3, 0.5 + 1.2j))
    assert np.allclose(displace(1).to_matrix(dims, squeezing = 0.5 + 1.2j, displacement = 0.5 + 1.2j), displace_matrix(3, 0.5 + 1.2j))
    params = {"squeezing": 0.5 + 1.2j, "displacement": 0.5 + 1.2j}
    assert np.allclose(squeeze(1).to_matrix(dims, params), squeeze_matrix(3, 0.5 + 1.2j))
    assert np.allclose(displace(1).to_matrix(dims, params), displace_matrix(3, 0.5 + 1.2j))
    with pytest.raises(Exception): squeeze(1).to_matrix(dims, displacement = 0.5 + 1.2j)
    with pytest.raises(Exception): displace(1).to_matrix(dims, squeeze = 0.5 + 1.2j)

    squeeze_params = squeeze(1).parameters
    print(squeeze_params)
    assert "squeezing" in squeeze_params
    assert squeeze_params["squeezing"] != ""

    displace_params = displace(1).parameters
    print(displace_params)
    assert "displacement" in displace_params
    assert displace_params["displacement"] != ""


def test_construction():
    prod = identity()
    assert np.allclose(prod.to_matrix(), identity_matrix(1))
    prod *= number(0)
    assert np.allclose(prod.to_matrix({0: 3}), number_matrix(3))
    sum = empty()
    assert np.allclose(sum.to_matrix(), zero_matrix(1))
    sum *= number(0)
    assert sum.degrees == []
    assert np.allclose(sum.to_matrix(), zero_matrix(1))
    sum += identity(1)
    assert sum.degrees == [1]
    assert np.allclose(sum.to_matrix({1: 3}), identity_matrix(3))
    sum *= number(1)
    assert np.allclose(sum.to_matrix({1: 3}), number_matrix(3))
    sum = empty()
    assert np.allclose(sum.to_matrix(), zero_matrix(1))
    sum -= identity(0)
    assert sum.degrees == [0]
    assert np.allclose(sum.to_matrix({0: 3}), -identity_matrix(3))
    ids = identities(3, 5)
    assert ids.degrees == [3, 4]
    assert np.allclose(ids.to_matrix({3: 3, 4: 3}), identity_matrix(3 * 3))
    canon = ids.copy().canonicalize()
    assert ids.degrees == [3, 4]
    assert canon.degrees == []
    assert canon.to_matrix() == identity_matrix(1)


def test_iteration():
    prod1 = position(1) * momentum(0)
    prod2 = number(0) * parity(0)
    sum = prod1 + prod2
    for p1, p2 in zip(sum, [prod1, prod2]):
        for t1, t2 in zip(p1, p2):
            assert t1 == t2
    sum_terms = 0
    prod_terms = 0
    for prod in sum:
        sum_terms += 1
        term_id = ""
        for op in prod:
            prod_terms += 1
            term_id += op.to_string(include_degrees = True)
        assert term_id == prod.term_id
    assert sum_terms == 2
    assert prod_terms == 4


def test_properties():

    prod1 = position(1) * momentum(0)
    prod2 = number(1) * parity(3)
    sum = prod1 + prod2
    assert prod1.degrees == [0, 1]
    assert prod2.degrees == [1, 3]
    assert sum.degrees == [0, 1, 3]
    assert prod1.min_degree == 0
    assert prod1.max_degree == 1
    assert prod2.min_degree == 1
    assert prod2.max_degree == 3
    assert sum.min_degree == 0
    assert sum.max_degree == 3

    dims = {0: 2, 1: 3, 2: 2, 3: 4}
    assert sum.term_count == 2
    assert prod1.ops_count == 2
    sum += prod1
    assert sum.term_count == 2
    prod1_mat = np.kron(identity_matrix(4), np.kron(position_matrix(3), momentum_matrix(2)))
    prod2_mat = np.kron(parity_matrix(4), np.kron(number_matrix(3), identity_matrix(2)))
    assert np.allclose(sum.to_matrix(dims), prod1_mat + prod1_mat + prod2_mat)

    prod1.dump()
    sum.dump()
    assert str(prod1) == "(1.000000+0.000000i) * momentum(0)position(1)"
    assert str(sum) == "(2.000000+0.000000i) * momentum(0)position(1) + (1.000000+0.000000i) * number(1)parity(3)"
    assert prod1.term_id == "momentum(0)position(1)"


def test_canonicalization():
    dims = {0: 2, 1: 3, 2: 2, 3: 4}
    all_degrees = [0, 1, 2, 3]

    # product operator
    for id_target in all_degrees:
        op = identity()
        expected = identity()
        for target in all_degrees:
            if target == id_target:
                op *= identity(target)
            elif target % 2 == 0:
                op *= parity(target)
                expected *= parity(target)
            else:
                op *= number(target)
                expected *= number(target)

        assert op != expected
        assert op.degrees == all_degrees
        op.canonicalize()
        assert op == expected
        assert op.degrees != all_degrees
        assert op.degrees == expected.degrees
        assert np.allclose(op.to_matrix(dims), expected.to_matrix(dims))

        op.canonicalize(set(all_degrees))
        assert op.degrees == all_degrees
        canon = canonicalized(op)
        assert op.degrees == all_degrees
        assert canon.degrees == expected.degrees

    # sum operator
    previous = empty()
    expected = empty()
    def check_expansion(got, want_degrees):
        canon = got.copy() # standard python behavior is for assignments not to copy
        term_with_missing_degrees = False
        for term in canon:
            if term.degrees != all_degrees:
                term_with_missing_degrees = True
        assert term_with_missing_degrees
        assert canon == got
        canon.canonicalize(want_degrees)
        assert canon != got
        assert canon.degrees == all_degrees
        for term in canon:
            assert term.degrees == all_degrees

    for id_target in all_degrees:
        term = identity()
        expected_term = identity()
        for target in all_degrees:
            if target == id_target:
                term *= identity(target)
            elif target & 2:
                term *= position(target)
                expected_term *= position(target)
            else:
                term *= momentum(target)
                expected_term *= momentum(target)
        previous += term
        expected += expected_term
        got = previous

        assert got != expected
        assert canonicalized(got) == expected
        assert got != expected
        got.canonicalize()
        assert got == expected
        assert got.degrees == expected.degrees
        assert np.allclose(got.to_matrix(dims), expected.to_matrix(dims))
        check_expansion(got, set(all_degrees))
        if id_target > 0: check_expansion(got, set())
        with pytest.raises(Exception):
            got.canonicalize(got.degrees[1:])


def test_trimming():
    all_degrees = [idx for idx in range(6)]
    dims = dict(((d, 2) for d in all_degrees))
    for _ in range(10):
        bit_mask = random.getrandbits(len(all_degrees))
        expected = empty()
        terms = [identity()] * len(all_degrees)
        # randomize order in which we add terms
        term_order = np.random.permutation(range(len(all_degrees)))
        for idx in range(len(all_degrees)):
            coeff = (bit_mask >> idx) & 1
            prod = identity(all_degrees[idx]) * float(coeff)
            if coeff > 0:
                expected += prod
            terms[term_order[idx]] = prod
        orig = empty()
        for term in terms:
            orig += term
        assert orig.term_count == len(all_degrees)
        assert orig.degrees == all_degrees
        orig.trim()
        assert orig.term_count < len(all_degrees)
        assert orig.term_count == expected.term_count
        assert orig.degrees == expected.degrees
        assert np.allclose(orig.to_matrix(dims), expected.to_matrix(dims))
        # check that our term map seems accurate
        for term in expected:
            orig += float(term.degrees[0]) * term
        assert orig.term_count == expected.term_count
        assert orig.degrees == expected.degrees
        for term in orig:
            assert term.evaluate_coefficient() == term.degrees[0] + 1.


def test_equality():
    prod1 = position(0) * momentum(0)
    prod2 = position(1) * momentum(1)
    prod3 = position(0) * momentum(1)
    prod4 = momentum(1) * position(0)
    sum = MatrixOperator(prod1)
    assert prod1 != prod2
    assert prod3 == prod4
    assert sum == prod1
    sum += prod3
    assert sum != prod1
    assert sum == (prod3 + prod1)
    sum += prod1
    assert sum != (prod3 + prod1)
    assert sum == (prod3 + 2. * prod1)
    assert sum != sum + 1.
    assert sum != identity(2) * sum
    dims = {0: 2, 1: 3, 2: 2, 3: 4}
    assert np.allclose(np.kron(identity_matrix(2), sum.to_matrix(dims)), (identity(2) * sum).to_matrix(dims))


def test_arithmetics():
    # basic tests for all arithmetic related bindings - 
    # more complex expressions are tested as part of the C++ tests
    dims = {0: 3, 1: 2}
    id = identity(0)
    sum = momentum(0) + position(1)
    sum_matrix = np.kron(position_matrix(2), identity_matrix(3)) +\
                 np.kron(identity_matrix(2), momentum_matrix(3))
    assert np.allclose(id.to_matrix(dims), identity_matrix(3))
    assert np.allclose(sum.to_matrix(dims), sum_matrix)

    # unary operators
    assert np.allclose((-id).to_matrix(dims), -1. * identity_matrix(3))
    assert np.allclose((-sum).to_matrix(dims), -1. * sum_matrix)
    assert np.allclose(id.to_matrix(dims), identity_matrix(3))
    assert np.allclose(sum.to_matrix(dims), sum_matrix)
    assert np.allclose((+id).canonicalize().to_matrix(), identity_matrix(1))
    assert np.allclose((+sum).canonicalize().to_matrix(dims), sum_matrix)
    assert np.allclose(id.to_matrix(dims), identity_matrix(3))

    # right-hand arithmetics
    assert np.allclose((id * 2.).to_matrix(dims), 2. * identity_matrix(3))
    assert np.allclose((sum * 2.).to_matrix(dims), 2. * sum_matrix)
    assert np.allclose((id * 2.j).to_matrix(dims), 2.j * identity_matrix(3))
    assert np.allclose((sum * 2.j).to_matrix(dims), 2.j * sum_matrix)
    assert np.allclose((sum * id).to_matrix(dims), sum_matrix)
    assert np.allclose((id * sum).to_matrix(dims), sum_matrix)
    assert np.allclose((id + 2.).to_matrix(dims), 3. * identity_matrix(3))
    assert np.allclose((sum + 2.).to_matrix(dims), sum_matrix + 2. * identity_matrix(2 * 3))
    assert np.allclose((id + 2.j).to_matrix(dims), (1. + 2.j) * identity_matrix(3))
    assert np.allclose((sum + 2.j).to_matrix(dims), sum_matrix + 2.j * identity_matrix(2 * 3))
    assert np.allclose((sum + id).to_matrix(dims), sum_matrix + identity_matrix(2 * 3))
    assert np.allclose((id + sum).to_matrix(dims), sum_matrix + identity_matrix(2 * 3))
    assert np.allclose((id - 2.).to_matrix(dims), -1. * identity_matrix(3))
    assert np.allclose((sum - 2.).to_matrix(dims), sum_matrix - 2. * identity_matrix(2 * 3))
    assert np.allclose((id - 2.j).to_matrix(dims), (1. - 2.j) * identity_matrix(3))
    assert np.allclose((sum - 2.j).to_matrix(dims), sum_matrix - 2.j * identity_matrix(2 * 3))
    assert np.allclose((sum - id).to_matrix(dims), sum_matrix - identity_matrix(2 * 3))
    assert np.allclose((id - sum).to_matrix(dims), identity_matrix(2 * 3) - sum_matrix)

    # in-place arithmetics
    term = id.copy()
    op = +sum
    term *= 2.
    op *= 2.
    assert np.allclose(term.to_matrix(dims), 2. * identity_matrix(3))
    assert np.allclose(op.to_matrix(dims), 2. * sum_matrix)
    term *= 0.5j
    op *= 0.5j
    assert np.allclose(term.to_matrix(dims), 1.j * identity_matrix(3))
    assert np.allclose(op.to_matrix(dims), 1.j * sum_matrix)
    op *= term
    assert np.allclose(op.to_matrix(dims), -1. * sum_matrix)

    op += 2.
    assert np.allclose(op.to_matrix(dims), -1. * sum_matrix + 2. * identity_matrix(2 * 3))
    op += term
    assert np.allclose(op.to_matrix(dims), -1. * sum_matrix + (2. + 1.j) * identity_matrix(2 * 3))
    op -= 2.
    assert np.allclose(op.to_matrix(dims), -1. * sum_matrix + 1.j * identity_matrix(2 * 3))
    op -= term
    assert np.allclose(op.to_matrix(dims), -1. * sum_matrix)

    # left-hand arithmetics
    assert np.allclose((2. * id).to_matrix(dims), 2. * identity_matrix(3))
    assert np.allclose((2. * sum).to_matrix(dims), 2. * sum_matrix)
    assert np.allclose((2.j * id).to_matrix(dims), 2.j * identity_matrix(3))
    assert np.allclose((2.j * sum).to_matrix(dims), 2.j * sum_matrix)
    assert np.allclose((2. + id).to_matrix(dims), 3. * identity_matrix(3))
    assert np.allclose((2. + sum).to_matrix(dims), sum_matrix + 2. * identity_matrix(2 * 3))
    assert np.allclose((2.j + id).to_matrix(dims), (1 + 2j) * identity_matrix(3))
    assert np.allclose((2.j + sum).to_matrix(dims), sum_matrix + 2.j * identity_matrix(2 * 3))
    assert np.allclose((2. - id).to_matrix(dims), identity_matrix(3))
    assert np.allclose((2. - sum).to_matrix(dims), 2. * identity_matrix(2 * 3) - sum_matrix)
    assert np.allclose((2.j - id).to_matrix(dims), (-1 + 2.j) * identity_matrix(3))
    assert np.allclose((2.j - sum).to_matrix(dims), 2.j * identity_matrix(2 * 3) - sum_matrix)


def test_term_distribution():
    op = empty()
    for target in range(7):
        op += identity(target)
    batches = op.distribute_terms(4)
    assert op.term_count == 7
    assert len(batches) == 4
    for idx in range(3):
        assert batches[idx].term_count == 2
    assert batches[3].term_count == 1
    sum = empty()
    for batch in batches:
        sum += batch
    assert sum == op

def op_definition(dim):
    return np.diag([(-1. + 0j)**i for i in range(dim)])

def define_ops(): 
    ElementaryOperator.define(
        "custom_parity1", [0],
        lambda dim: np.diag([(-1. + 0j)**i for i in range(dim)]))
    ElementaryOperator.define(
        "custom_parity2", [0],
        lambda dim: np.diag([(-1. + 0j)**i for i in range(dim)]))

def test_custom_operators():
    define_ops()
    custom1 = ElementaryOperator("custom_parity1", [1])
    print(custom1.to_matrix({1 : 5}))
    custom2 = ElementaryOperator("custom_parity2", [1])
    print(custom2.to_matrix({1 : 5}))
    #assert False


# Run with: pytest -rP
if __name__ == "__main__":
    pytest.main(["-rP"])