//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/EmPhysicsList.cc
//---------------------------------------------------------------------------//
#include "EmPhysicsList.hh"

#include <memory>

#include "celeritas/Quantities.hh"

#include "detail/EmStandardPhysics.hh"
#include "detail/OpticalPhysics.hh"
#include "detail/PhysicsListUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with physics options.
 */
EmPhysicsList::EmPhysicsList(Options const& options)
{
    using ClhepLen = Quantity<units::ClhepTraits::Length, double>;

    this->SetVerboseLevel(options.verbose);
    this->SetDefaultCutValue(
        native_value_to<ClhepLen>(options.default_cutoff).value());

    // Celeritas-supported EM Physics
    detail::emplace_physics<detail::EmStandardPhysics>(*this, options);

    if (options.optical)
    {
        // Celeritas-supported Optical Physics
        detail::emplace_physics<detail::OpticalPhysics>(*this, options.optical);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
