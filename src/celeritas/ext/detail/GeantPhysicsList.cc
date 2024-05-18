//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantPhysicsList.cc
//---------------------------------------------------------------------------//
#include "GeantPhysicsList.hh"

#include <memory>

#include "celeritas/Quantities.hh"

#include "CelerEmStandardPhysics.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with physics options.
 */
GeantPhysicsList::GeantPhysicsList(Options const& options)
{
    using ClhepLen = Quantity<units::ClhepTraits::Length, double>;

    this->SetVerboseLevel(options.verbose);
    this->SetDefaultCutValue(
        native_value_to<ClhepLen>(options.default_cutoff).value());

    // Celeritas-supported EM Physics
    auto em_standard = std::make_unique<CelerEmStandardPhysics>(options);
    RegisterPhysics(em_standard.release());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
