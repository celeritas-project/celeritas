//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/PolishedRoughnessExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
struct PolishedRoughnessExecutor
{
    Real3 const& normal;

    template<class Engine>
    CELER_FUNCTION Real3 operator()(Engine&) const
    {
        return normal;
    }
};

struct PolishedRoughnessExecutorBuilder
{
    CELER_FUNCTION PolishedRoughnessExecutor operator()(
        SurfaceModelView const&, Real3 const&, Real3 const& normal) const
    {
        return PolishedRoughnessExecutor{normal};
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
