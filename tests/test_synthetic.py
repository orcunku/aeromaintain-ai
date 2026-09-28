import numpy as np

from aeromaintain.synthetic.generator import (
    COMPONENT_TYPES,
    create_components,
    create_fleet,
)


def test_fleet_size():

    rng = np.random.default_rng(42)

    fleet = create_fleet(
        n_aircraft=10,
        rng=rng,
    )

    assert len(fleet) == 10


def test_aircraft_ids_are_unique():

    rng = np.random.default_rng(42)

    fleet = create_fleet(
        n_aircraft=10,
        rng=rng,
    )

    assert fleet["aircraft_id"].is_unique


def test_component_count():

    rng = np.random.default_rng(42)

    fleet = create_fleet(
        n_aircraft=10,
        rng=rng,
    )

    components = create_components(
        fleet=fleet,
        rng=rng,
    )

    expected = (
        len(fleet)
        * len(COMPONENT_TYPES)
    )

    assert len(components) == expected


def test_component_ids_are_unique():

    rng = np.random.default_rng(42)

    fleet = create_fleet(
        n_aircraft=10,
        rng=rng,
    )

    components = create_components(
        fleet=fleet,
        rng=rng,
    )

    assert components[
        "component_id"
    ].is_unique


def test_utilization_factor_range():

    rng = np.random.default_rng(42)

    fleet = create_fleet(
        n_aircraft=100,
        rng=rng,
    )

    assert (
        fleet["utilization_factor"]
        .between(0.65, 1.40)
        .all()
    )


def test_component_susceptibility_range():

    rng = np.random.default_rng(42)

    fleet = create_fleet(
        n_aircraft=10,
        rng=rng,
    )

    components = create_components(
        fleet=fleet,
        rng=rng,
    )

    assert (
        components[
            "susceptibility_factor"
        ]
        .between(
            0.70,
            1.50,
        )
        .all()
    )