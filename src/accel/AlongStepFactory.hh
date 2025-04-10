//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/AlongStepFactory.hh
//! \brief Along-step factory interface and definitions
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>
#include <G4ThreeVector.hh>

#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/global/ActionInterface.hh"

class G4LogicalVolume;

namespace celeritas
{
struct ImportData;
struct RZMapFieldInput;
struct CylMapFieldInput;
class CutoffParams;
class FluctuationParams;
class GeoMaterialParams;
class MaterialParams;
class ParticleParams;
class PhysicsParams;

namespace inp
{
struct UniformField;
}

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
 * `celeritas/alongstep`.
 */
class AlongStepFactoryInterface
{
  public:
    //!@{
    //! \name Type aliases
    using argument_type = AlongStepFactoryInput const&;
    using result_type = std::shared_ptr<CoreStepActionInterface const>;
    //!@}

  public:
    virtual ~AlongStepFactoryInterface() = default;

    // Emit an along-step action
    virtual result_type operator()(argument_type input) const = 0;

  protected:
    AlongStepFactoryInterface() = default;
    CELER_DEFAULT_COPY_MOVE(AlongStepFactoryInterface);
};

//---------------------------------------------------------------------------//
/*!
 * Create an along-step method for a uniform (or zero) field.
 *
 * The constructor is a lazily evaluated function that must return the field
 * definition and driver configuration. If unspecified, the field is zero.
 *
 * \todo Add a helper function to build from a Geant4 field manager or the new
 * \c G4FieldSetup
 */
class UniformAlongStepFactory final : public AlongStepFactoryInterface
{
  public:
    //!@{
    //! \name Type aliases
    using FieldInput = inp::UniformField;
    using FieldFunction = std::function<FieldInput()>;
    using VecVolume = std::vector<G4LogicalVolume const*>;
    using VecVolumeFunction = std::function<VecVolume()>;
    //!@}

  public:
    //! Construct with no field (linear propagation)
    UniformAlongStepFactory() = default;

    // Construct with a function to return the field strength
    explicit UniformAlongStepFactory(FieldFunction f);

    // Construct with field strength and volumes where field is present
    UniformAlongStepFactory(FieldFunction f, VecVolumeFunction volumes);

    // Emit an along-step action
    result_type operator()(argument_type input) const final;

    // Get the field params (used for converting to celeritas::inp)
    FieldInput get_field() const;

    // Get the volumes where field is present
    VecVolume get_volumes() const;

  private:
    FieldFunction get_field_;
    VecVolumeFunction get_volumes_;
};

//---------------------------------------------------------------------------//
/*!
 * Create an along-step method for a two-dimensional (r-z in the cylindical
 * coordinate system) map field (RZMapField).
 */
class RZMapFieldAlongStepFactory final : public AlongStepFactoryInterface
{
  public:
    //!@{
    //! \name Type aliases
    using RZMapFieldFunction = std::function<RZMapFieldInput()>;
    //!@}

  public:
    // Construct with a function to return RZMapFieldInput
    explicit RZMapFieldAlongStepFactory(RZMapFieldFunction f);

    // Emit an along-step action
    result_type operator()(argument_type input) const final;

    // Get the field params (used for converting to celeritas::inp)
    RZMapFieldInput get_field() const;

  private:
    RZMapFieldFunction get_fieldmap_;
};

//---------------------------------------------------------------------------//
/*!
 * Create an along-step method for a three-dimensional (r-phi-z in the
 * cylindical coordinate system) map field (CylMapField).
 */
class CylMapFieldAlongStepFactory final : public AlongStepFactoryInterface
{
  public:
    //!@{
    //! \name Type aliases
    using CylMapFieldFunction = std::function<CylMapFieldInput()>;
    //!@}

  public:
    // Construct with a function to return CylMapFieldInput
    explicit CylMapFieldAlongStepFactory(CylMapFieldFunction f);

    // Emit an along-step action
    result_type operator()(argument_type input) const final;

    // Get the field params (used for converting to celeritas::inp)
    CylMapFieldInput get_field() const;

  private:
    CylMapFieldFunction get_fieldmap_;
};
//---------------------------------------------------------------------------//
}  // namespace celeritas
