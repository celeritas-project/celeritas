# Celeritas app components

Each component is just an abstract box: its implementation could be a single
object, collection of objects, or possibly even combination with another box.
The components should be basic and as independent as possible. They are each
divided into "parameters", run-time invariants given by the user, and "states",
which are unique to a track.

## Navigation

Responsible for mapping all points in space to non-overlapping logical volumes.

(Shift nomenclature: this is called Geometry.)

### Parameters

Stateless geometry definition constructed by GDML: the CSG representation of
the universe.

### State

- Position
- Direction
- Logical volume ID

Particular implementations may have additional state, such as the "touchable
history" (VecGeom volume hierarchy location).

## Materials

Describes the atomic composition of a particular volume space. Each material is
assumed to be completely homogeneous.

### Parameters

A list of individual materials, each of which has elements/nuclides and
mass/number densities.

### State

- Material ID

## Random number generator

Supplies a random bitstream (or stream of uniformly sampled floating point
numbers).

### Parameters

RNG type, seeding mechanism for each event/track.

### State

Opaque RNG state.

## Physics

Physical properties 

### Parameters

- Which particle types are being simulated,
- Data about each particle type (mass, charge, etc.)
- (??) Which physics models, and for what particle types/energy ranges they
  apply

### State

- Particle type ID
- Energy

Properties such as charge and rest mass are *derivative* from the particle ID.
Also derivative would be the list of physics models that apply to the track
based on the type and energy, but it may be necessary to have that list (or an
equivalent mask) be part of the state in order for the individual physics
models to have state.

Momentum is a derivative quantity of the geometry (direction), energy, 

## Stepper

Propagates particle path through magnetic fields.

### Parameters

- Magnetic field definition, etc.
- Stepper model selection (Runge-Kutta etc)

### State

A derivative property (from particle type and spatial position) would be which
fields at what magnitudes apply to the track.

## Introspection

Track identification: what event and parent track it came from, etc.;
cumulative time since initial event.

### Parameters

None?

### State

- Track ID
- Time
- Parent track ID (?)
- Event ID (?)

Depending on implementation, the parent track ID and event ID might be
derivative quantities.

## Detectors

Calculate/store/write out energy deposition (and other quantities as necessary)
in particular volumes of space.

(Shift nomenclature: tallies)

### Parameters

- Applicable regions of phase space (volume ID, particle type, energy)

### State

???

(Possible example: energy when entering sensitive region, so the energy when
exiting the region can be subtracted and tallied.)

## Secondaries

Create and buffer secondary particles created by a track.

### Parameters

??? Maybe none if this functionality is globally shared across tracks.
