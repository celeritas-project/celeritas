//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/AlongStepFactory.hh
//! \brief Along-step factory interface and definitions
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <G4ThreeVector.hh>

#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/geo/GeoParamsFwd.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
struct ImportData;
class CutoffParams;
class FluctuationParams;
class GeoMaterialParams;
class MaterialParams;
class ParticleParams;
class PhysicsParams;

//---------------------------------------------------------------------------//
/*!
 * Input argument to the AlongStepFactory interface.
 *
 * When passed to a factory instance, all member data will be set (so the
 * instance will be 'true').
 *
 * Most of these classes have been forward-declared because they simply need to
 * be passed along to another class's constructor.
 */
struct AlongStepFactoryInput
{
    ActionId action_id;

    std::shared_ptr<GeoParams const> geometry;
    std::shared_ptr<MaterialParams const> material;
    std::shared_ptr<GeoMaterialParams const> geomaterial;
    std::shared_ptr<ParticleParams const> particle;
    std::shared_ptr<CutoffParams const> cutoff;
    std::shared_ptr<PhysicsParams const> physics;
    std::shared_ptr<ImportData const> imported;

    //! True if all data is assigned
    explicit operator bool() const
    {
        return action_id && geometry && material && geomaterial && particle
               && cutoff && physics && imported;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Helper class for emitting an AlongStep action.
 *
 * Currently Celeritas accepts a single along-step action (i.e., the same
 * stepper is used for both neutral and charged particles, across all energies
 * and regions of the problem). The along-step action is a single GPU
 * kernel that combines the field stepper selection, the magnetic field,
 * slowing-down calculation, multiple scattering, and energy loss fluctuations.
 *
 * The factory will be called from the thread that initializes \c SharedParams.
 * Instead of a daughter class, you can provide any function-like object that
 * has the same interface.
 *
 * Celeritas provides a few "default" configurations of along-step actions in
 * `celeritas/global/alongstep`.
 */
class AlongStepFactoryInterface
{
  public:
    //!@{
    //! \name Type aliases
    using argument_type = AlongStepFactoryInput const&;
    using result_type = std::shared_ptr<ExplicitActionInterface const>;
    //!@}

  public:
    virtual ~AlongStepFactoryInterface() = default;

    // Emit an along-step action
    virtual result_type operator()(argument_type input) const = 0;
};

//---------------------------------------------------------------------------//
/*!
 * Create an along-step method for a uniform (or zero) field.
 *
 * The constructor is a lazily evaluated function that must return the field
 * vector in native Geant4 units.  If unspecified, the field is zero.
 */
class UniformAlongStepFactory : public AlongStepFactoryInterface
{
  public:
    //!@{
    //! \name Type aliases
    using FieldFunction = std::function<G4ThreeVector()>;
    //!@}

  public:
    //! Construct with no field (linear propagation)
    UniformAlongStepFactory() = default;

    // Construct with a function to return the field strength
    explicit UniformAlongStepFactory(FieldFunction f);

    // Emit an along-step action
    result_type operator()(argument_type input) const final;

  private:
    FieldFunction get_field_;
};

//---------------------------------------------------------------------------//
/*!
 * Create an along-step method for a two-dimensional (r-z in the cylindical
 * coordinate system) map field (RZMapField).
 */
class RZMapFieldAlongStepFactory : public AlongStepFactoryInterface
{
  public:
    // Construct with a two-dimensional (r-z) field map with input json file
    explicit RZMapFieldAlongStepFactory(std::string filename);

    // Emit an along-step action
    result_type operator()(argument_type input) const final;

  private:
    RZMapFieldInput field_map_;
};
//---------------------------------------------------------------------------//
}  // namespace celeritas
