//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/AlongStepFactory.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
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

    std::shared_ptr<const GeoParams>         geometry;
    std::shared_ptr<const MaterialParams>    material;
    std::shared_ptr<const GeoMaterialParams> geomaterial;
    std::shared_ptr<const ParticleParams>    particle;
    std::shared_ptr<const CutoffParams>      cutoff;
    std::shared_ptr<const PhysicsParams>     physics;
    std::shared_ptr<const ImportData>        imported;

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
    using argument_type = const AlongStepFactoryInput&;
    using result_type = std::shared_ptr<const ExplicitActionInterface>;
    //!@}

  public:
    virtual ~AlongStepFactoryInterface() = default;

    // Emit an along-step action
    virtual result_type operator()(argument_type input) const = 0;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
