//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/PhysicsData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericGridData.hh"

#include "Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//

struct PhysicsParamsScalars
{
    explicit CELER_FUNCTION operator bool() const { return false; }
};

template<Ownership W, MemSpace M>
struct PhysicsParamsData
{
    PhysicsParamsScalars scalars;

    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(scalars);
    }

    template<Ownership W2, MemSpace M2>
    PhysicsParamsData<W, M>& operator=(PhysicsParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        this->scalars = other.scalars;
        return *this;
    }
};

template<Ownership W, MemSpace M>
struct PhysicsStateData
{
    explicit CELER_FUNCTION operator bool() const { return false; }

    CELER_FUNCTION size_type size() const { return 0; }

    template<Ownership W2, MemSpace M2>
    PhysicsStateData<W, M>& operator=(PhysicsStateData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        return *this;
    }
};

template<MemSpace M>
inline void resize(PhysicsStateData<Ownership::value, M>* state, size_type size)
{
    CELER_EXPECT(state);
    CELER_EXPECT(size > 0);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
