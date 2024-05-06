//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ProtoInterface.cc
//---------------------------------------------------------------------------//
#include "ProtoInterface.hh"

#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"

#include "detail/InputBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Construct all universes.
 */
OrangeInput build_input(Tolerance<> const& tol, ProtoInterface const& global)
{
    ScopedProfiling profile_this{"build-orange-geo"};
    ScopedMem record_mem("orangeinp::build_input");
    ScopedTimeLog scoped_time;

    OrangeInput result;
    detail::ProtoMap const protos{global};
    CELER_ASSERT(protos.find(&global) == orange_global_universe);
    detail::InputBuilder builder(&result, tol, protos);
    for (auto uid : range(UniverseId{protos.size()}))
    {
        protos.at(uid)->build(builder);
    }

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
