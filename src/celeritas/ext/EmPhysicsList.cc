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
    auto em_standard = std::make_unique<detail::EmStandardPhysics>(options);
    RegisterPhysics(em_standard.release());

    if (options.optical)
    {
        // Celeritas-supported Optical Physics
        auto optical_physics
            = std::make_unique<detail::OpticalPhysics>(options.optical);
        RegisterPhysics(optical_physics.release());
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
