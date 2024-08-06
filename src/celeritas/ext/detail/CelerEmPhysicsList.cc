//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/CelerEmPhysicsList.cc
//---------------------------------------------------------------------------//
#include "CelerEmPhysicsList.hh"

#include <memory>

#include "celeritas/Quantities.hh"

#include "CelerEmStandardPhysics.hh"
#include "CelerOpticalPhysics.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with physics options.
 */
CelerEmPhysicsList::CelerEmPhysicsList(Options const& options)
{
    using ClhepLen = Quantity<units::ClhepTraits::Length, double>;

    this->SetVerboseLevel(options.verbose);
    this->SetDefaultCutValue(
        native_value_to<ClhepLen>(options.default_cutoff).value());

    // Celeritas-supported EM Physics
    auto em_standard = std::make_unique<CelerEmStandardPhysics>(options);
    RegisterPhysics(em_standard.release());

    // Celeritas-supported Optical Physics
    if (options.optical_options)
    {
        auto optical_physics = std::make_unique<CelerOpticalPhysics>(
            options.optical_options.value());
        RegisterPhysics(optical_physics.release());
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
