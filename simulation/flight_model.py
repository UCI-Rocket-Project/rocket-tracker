# PTR flight model by Hudson (modified a bit by Eric)

from rocketpy import (
    LiquidMotor,
    MassFlowRateBasedTank,
    Fluid,
    Rocket,
    Flight,
    Environment,
    CylindricalTank,
)

## Environment
date_info = (2023, 4, 29, 23)  # Launch Date, Hour given in UTC time

geodetic = [35.347104, -117.808953, 620]

EnvGFS = Environment(date=date_info, latitude=geodetic[0], longitude=geodetic[1], elevation=geodetic[2], datum='NAD83')
EnvGFS.set_atmospheric_model(type="Windy", file="GFS")

EnvTrue = EnvGFS # set which env to use
EnvTrue.maxExpectedHeight = 10000

## Fluids
GasTemp = 305 #K
COPVTankPres = 2.8958e+7 #Pa
LOXTankPres = 3465498.069 #Pa
LNGTankPres = 2768107.16 #Pa
N2GasCons = 296.8 #J/kgK

DensityScalar = .95
LOX = Fluid(name="LOX", density=DensityScalar*1140) # maybe decrease cuz we likely wont have ideal dens
LNG = Fluid(name="LNG", density=DensityScalar*422) # maybe decrease cuz we likely wont have ideal dens
LOXTankPressurizingGas = Fluid(name="N2", density=((LOXTankPres)/(GasTemp*N2GasCons)))
LNGTankPressurizingGas = Fluid(name="N2", density=((LNGTankPres)/(GasTemp*N2GasCons)))
PressurizingGas = Fluid(name="N2", density=((COPVTankPres)/(GasTemp*N2GasCons)))

## Tank Geom
PropTankVol = 0.0245806
COPVVol = .009

LOX_tank_geometry = CylindricalTank(0.0986, 0.868, spherical_caps=True) # there are weird edge cases where the tank caps will break.
LNG_tank_geometry = CylindricalTank(0.0986, 0.868, spherical_caps=True)
#rad = 3.885 in
#cyllengrth = 26.5 in, 34.22in
COPV_geometry = CylindricalTank(0.07493, 0.56134, spherical_caps=True)

PropScalar = 1
burntiempo = 13.833

## LOX mass/time
# LOX scalar default .770326
# liquid mass 21.586, 28.051
# using approximate tank mass (not full) based on burn time and mdot
LOXScalar = .770326*PropScalar
LOX_mass_tank = MassFlowRateBasedTank(
    name = "LOXMassFlowRateBasedTank",
    geometry = LOX_tank_geometry,
    flux_time = burntiempo,

    gas=LOXTankPressurizingGas,
    initial_gas_mass=0,

    liquid = LOX,
    initial_liquid_mass = LOXScalar*PropTankVol*LOX.density,

    liquid_mass_flow_rate_in = 0,
    liquid_mass_flow_rate_out = (LOXScalar*PropTankVol*LOX.density)/(burntiempo+.007),
    gas_mass_flow_rate_in = 0,
    gas_mass_flow_rate_out = 0,

    discretize=100,
)

## LNG mass/time
# LNG scalar default .717578
# liquid mass 7.4434508, 10.378
# using approximate tank mass (not full) based on burn time and mdot
# not considering residual LNG
LNGScalar = .717578*PropScalar
LNG_mass_tank = MassFlowRateBasedTank(
    name = "LNGMassFlowRateBasedTank",
    geometry = LNG_tank_geometry,
    flux_time = burntiempo,

    gas=LNGTankPressurizingGas,
    initial_gas_mass=0,

    liquid = LNG,
    initial_liquid_mass = LNGScalar*PropTankVol*LNG.density,

    liquid_mass_flow_rate_in = 0,
    liquid_mass_flow_rate_out = (LNGScalar*PropTankVol*LNG.density)/(burntiempo+.007),
    gas_mass_flow_rate_in = 0,
    gas_mass_flow_rate_out = 0,

    discretize=100,
)

COPV = MassFlowRateBasedTank(
    name = "COPVMassFlowRateBasedTank",
    geometry = COPV_geometry,
    flux_time = burntiempo,

    gas=LNGTankPressurizingGas,
    initial_gas_mass=(COPVVol*PressurizingGas.density)/2,

    liquid = LNGTankPressurizingGas,
    initial_liquid_mass = 0,

    liquid_mass_flow_rate_in = 0,
    liquid_mass_flow_rate_out = 0,
    gas_mass_flow_rate_in = 0,
    gas_mass_flow_rate_out = 0,
    # making mdot 0 cuz this contributes to mdot total which affect exhaust vel. instead dividing initial gas mass by 2
    discretize=100,
)
#(COPVVol*PressurizingGas.density)/(burntiempo+.007) in case ya wanna add mdot back

## Engine
# DEFINING ORIGIN AS Center of dry mass
PTE = LiquidMotor(
    thrust_source="simulation/flight_model_data/USETHIS_EstimatedFlightThrustOverTime.csv",
    reshape_thrust_curve=(False),
    center_of_dry_mass_position=1.88976,
    dry_inertia=(0, 0, 0),
    dry_mass=51.0472853,
    burn_time=burntiempo,
    nozzle_radius=0.0523,
    nozzle_position=0,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)
#for reshape, replace False with (burn time, impulse) tuple

PTE.add_tank(LNG_mass_tank, position=0.77792+0.434) #dist from nozzle to bottom of tank + half tank height
PTE.add_tank(LOX_mass_tank, position=2.0002+0.434)
PTE.add_tank(COPV, position=3.50774+0.28067)


## Launch Vehicle
PTR = Rocket(
    radius=0.147066,
    mass=91.777-PTE.dry_mass,
    inertia=(208, 208, 1.26),
    power_off_drag="simulation/flight_model_data/drag.csv",
    power_on_drag="simulation/flight_model_data/drag.csv",
    center_of_mass_without_motor=3.005328,
    coordinate_system_orientation="tail_to_nose",
)
# test mass with wet mass

PTR.add_motor(PTE, position=0)
PTR.add_nose(length=1.143, kind="vonKarman", position=4.52628)
PTR.add_trapezoidal_fins(
    n=3,
    root_chord=0.3048,
    tip_chord=0.0850392,
    span=0.2450592,
    position=0.314452,
    cant_angle=0,
)
PTR.set_rail_buttons(lower_button_position=1.2, upper_button_position=3.75, angular_position=60)

## Flight
test_flight = Flight(
    rocket=PTR,
    environment=EnvTrue,
    rail_length=18.288,
    inclination=90,
    heading=0,
    max_time_step=0.05,
    terminate_on_apogee=False,
)