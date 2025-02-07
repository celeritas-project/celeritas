//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/PhysicsListUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VModularPhysicsList.hh>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create a suitably owned physics class and add it to the physics list.
 */
template<class PL, class... T>
void emplace_physics(G4VModularPhysicsList& list, T&&... args)
{
    // Construct physics
    auto phys = std::make_unique<PL>(std::forward<T>(args)...);

    // Register, passing ownership to the list
    list.RegisterPhysics(phys.release());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
