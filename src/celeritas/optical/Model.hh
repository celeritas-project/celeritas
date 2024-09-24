//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Model.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/sys/ActionInterface.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
class CoreParams;
template<MemSpace M>
class CoreState;

namespace detail
{
class MfpBuilder;
}

//---------------------------------------------------------------------------//
/*!
 * Base class for discrete, volumetric optical models.
 *
 * For optical physics, there is precisely one particle (optical photons)
 * and one energy range (optical wavelengths), so only models and no processes
 * are used in optical physics.
 */
class Model : public StepActionInterface<CoreParams, CoreState>,
              public ConcreteAction
{
  public:
    // Construct with defaults
    Model(ActionId id, std::string const& label, std::string const& description)
        : ConcreteAction(id, label, description)
    {
    }

    // Virtual destructor for polymorphic deletion
    virtual ~Model() = default;

    // Action order for optical models is always post-step
    StepActionOrder order() const override final
    {
        return StepActionOrder::post;
    }

    // Build mean free path grids for all optical materials
    virtual void build_mfps(detail::MfpBuilder build) const = 0;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
