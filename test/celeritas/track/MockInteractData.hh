//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/MockInteractData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/data/Collection.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template<Ownership W, MemSpace M>
struct MockInteractData
{
    StateCollection<size_type, W, M> num_secondaries;
    StateCollection<char, W, M> alive;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !num_secondaries.empty()
               && alive.size() == num_secondaries.size();
    }

    //! State size
    CELER_FUNCTION size_type size() const { return num_secondaries.size(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    MockInteractData& operator=(MockInteractData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        this->num_secondaries = other.num_secondaries;
        this->alive = other.alive;

        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
